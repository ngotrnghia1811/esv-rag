"""
Two-stage ESV-RAG training pipeline (§3.7).

Stage 1 — Trace Generation:
    Curate complex questions from BREAK / Monaco,
    run MCTS to produce reasoning trajectories,
    serialise SFT + RL training examples (6× data multiplication).

Stage 2a — Supervised Fine-Tuning:
    Behavioural cloning on successful traces (L_SFT).

Stage 2b — PPO Training:
    Refine policy through environmental interaction via OpenRLHF (L_PPO).

Hardware target: 4× NVIDIA H100 GPUs, DeepSpeed ZeRO-3, Ray cluster.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import ESVConfig, TrainConfig
from .state import State
from .generator import Generator
from .retriever import RetrieverClient
from .planner import MCTSPlanner
from .rewards import ESVRewardCalculator

logger = logging.getLogger(__name__)


# ===========================================================================
# Data structures
# ===========================================================================

@dataclass
class ReasoningTrace:
    """One MCTS reasoning trajectory."""
    question: str
    action_sequence: List[str]
    states: List[Dict[str, Any]]      # serialised State at each step
    rewards: List[float]
    final_answer: Optional[str]
    correct: bool
    complexity_score: float


@dataclass
class SFTExample:
    """Single supervised fine-tuning example."""
    prompt: str
    response: str
    action: str
    reward: float


@dataclass
class RLExample:
    """Full trajectory for PPO training."""
    prompt: str
    trajectory: List[Dict]
    total_reward: float
    terminal_reward: float


# ===========================================================================
# Stage 1 — Data curation and trace generation
# ===========================================================================

class DataCurator:
    """
    Curate complex questions from BREAK and Monaco datasets.

    Only questions with complexity_score ≥ threshold are retained.
    """

    def __init__(self, config: TrainConfig):
        self.config = config

    def load_break(self, path: str) -> List[Dict]:
        """Load BREAK dataset (QDMR decomposition format)."""
        questions = []
        with open(path) as f:
            for line in f:
                obj = json.loads(line.strip())
                score = self._complexity_score(obj)
                if score >= self.config.complexity_threshold:
                    questions.append({
                        "question":    obj.get("question_text", obj.get("question", "")),
                        "answer":      obj.get("answer", ""),
                        "complexity":  score,
                        "source":      "break",
                        "decomposition": obj.get("decomposition", []),
                    })
        logger.info("BREAK: retained %d / loaded questions", len(questions))
        return questions

    def load_monaco(self, path: str) -> List[Dict]:
        """Load Monaco multi-document QA dataset."""
        questions = []
        with open(path) as f:
            data = json.load(f)
        for item in data:
            score = self._complexity_score(item, default_complexity=0.7)
            questions.append({
                "question":   item.get("question", ""),
                "answer":     item.get("answer", ""),
                "complexity": score,
                "source":     "monaco",
                "documents":  item.get("documents", []),
            })
        logger.info("Monaco: retained %d questions", len(questions))
        return questions

    def curate(self, break_path: Optional[str] = None,
               monaco_path: Optional[str] = None) -> List[Dict]:
        """Merge, filter, and truncate to num_curated_questions."""
        pool: List[Dict] = []
        if break_path and Path(break_path).exists():
            pool.extend(self.load_break(break_path))
        if monaco_path and Path(monaco_path).exists():
            pool.extend(self.load_monaco(monaco_path))

        # Sort by complexity descending, then sample
        pool.sort(key=lambda x: x["complexity"], reverse=True)
        pool = pool[: self.config.num_curated_questions]
        random.shuffle(pool)
        logger.info("Curated %d questions total", len(pool))
        return pool

    @staticmethod
    def _complexity_score(obj: Dict, default_complexity: float = 0.5) -> float:
        if "complexity" in obj:
            return float(obj["complexity"])
        # Estimate from decomposition length or question word count
        if "decomposition" in obj:
            return min(1.0, len(obj["decomposition"]) / 6.0)
        words = len(obj.get("question_text", obj.get("question", "")).split())
        return min(1.0, words / 30.0)


class TraceGenerator:
    """
    Run MCTS on curated questions to generate reasoning traces.
    Applies 6× data multiplication: 5 SFT examples + 1 RL example per trace.
    """

    def __init__(self, planner: MCTSPlanner, config: TrainConfig):
        self.planner = planner
        self.config  = config
        self.reward_calc = ESVRewardCalculator()

    def generate_traces(self, questions: List[Dict],
                         output_dir: str) -> Tuple[List[SFTExample], List[RLExample]]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        sft_examples: List[SFTExample] = []
        rl_examples:  List[RLExample]  = []

        for i, q_item in enumerate(questions):
            logger.info("Generating trace %d/%d: %s",
                        i + 1, len(questions), q_item["question"][:60])
            try:
                result = self.planner.run(q_item["question"])
                trace  = self._build_trace(q_item, result)

                sft = self._build_sft_examples(trace)
                rl  = self._build_rl_example(trace)

                sft_examples.extend(sft)
                rl_examples.append(rl)

                # Save incrementally
                self._save_trace(trace, out / f"trace_{i:05d}.json")

            except Exception as exc:
                logger.error("Trace generation failed for question %d: %s", i, exc)

        self._save_dataset(sft_examples, out / "sft_examples.jsonl")
        self._save_dataset(rl_examples,  out / "rl_examples.jsonl")

        logger.info("Generated %d SFT + %d RL examples from %d questions",
                    len(sft_examples), len(rl_examples), len(questions))
        return sft_examples, rl_examples

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_trace(self, q_item: Dict, result) -> ReasoningTrace:
        state = result.final_state
        answer_correct = self._check_correctness(state, q_item.get("answer", ""))
        return ReasoningTrace(
            question=q_item["question"],
            action_sequence=result.action_sequence,
            states=[s.model_dump() if hasattr(s, "model_dump") else {}
                    for s in (state.reasoning_history or [])],
            rewards=[self.reward_calc.calculate_reward(state)],
            final_answer=state.main_answer,
            correct=answer_correct,
            complexity_score=q_item.get("complexity", 0.5),
        )

    @staticmethod
    def _check_correctness(state: State, gold: str) -> bool:
        if not state.main_answer or not gold:
            return False
        pred_tokens = set(state.main_answer.lower().split())
        gold_tokens = set(gold.lower().split())
        if not gold_tokens:
            return False
        overlap = len(pred_tokens & gold_tokens) / len(gold_tokens)
        return overlap >= 0.5

    def _build_sft_examples(self, trace: ReasoningTrace) -> List[SFTExample]:
        """Create one SFT example per reasoning step (up to 5)."""
        examples: List[SFTExample] = []
        q = trace.question
        for i, action in enumerate(trace.action_sequence[:5]):
            prompt = (f"Question: {q}\n"
                      f"Reasoning history: {trace.states[:i]}\n"
                      f"Next action:")
            response = action
            reward   = trace.rewards[0] if trace.rewards else 0.0
            examples.append(SFTExample(prompt=prompt, response=response,
                                        action=action, reward=reward))
        return examples

    def _build_rl_example(self, trace: ReasoningTrace) -> RLExample:
        prompt = f"Question: {trace.question}\nReasoning:"
        terminal_r = ESVRewardCalculator.terminal_reward(
            correct=trace.correct,
            verified=bool(trace.action_sequence and "V" in trace.action_sequence),
            timeout=False,
        )
        return RLExample(
            prompt=prompt,
            trajectory=[{"action": a} for a in trace.action_sequence],
            total_reward=sum(trace.rewards),
            terminal_reward=terminal_r,
        )

    @staticmethod
    def _save_trace(trace: ReasoningTrace, path: Path) -> None:
        path.write_text(json.dumps(asdict(trace), indent=2))

    @staticmethod
    def _save_dataset(items: List, path: Path) -> None:
        with open(path, "w") as f:
            for item in items:
                f.write(json.dumps(asdict(item)) + "\n")


# ===========================================================================
# Stage 2 — Training
# ===========================================================================

class SFTTrainer:
    """
    Supervised Fine-Tuning (behavioural cloning) on MCTS traces.

    Uses HuggingFace Trainer with DeepSpeed ZeRO-3 for distributed training
    across 4× H100 GPUs.
    """

    def __init__(self, config: TrainConfig):
        self.config = config

    def train(self, sft_examples_path: str) -> None:
        import torch
        from datasets import load_dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            DataCollatorForLanguageModeling,
        )

        logger.info("Loading SFT examples from %s", sft_examples_path)
        dataset = load_dataset("json", data_files=sft_examples_path, split="train")

        tokenizer = AutoTokenizer.from_pretrained(self.config.sft_base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.config.sft_base_model,
            torch_dtype=torch.bfloat16,
        )

        def tokenize(batch):
            texts = [f"{p} {r}" for p, r in zip(batch["prompt"], batch["response"])]
            return tokenizer(texts, truncation=True, max_length=2048, padding=False)

        dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

        args = TrainingArguments(
            output_dir=self.config.sft_output_dir,
            num_train_epochs=self.config.sft_epochs,
            per_device_train_batch_size=self.config.sft_batch_size // self.config.num_gpus,
            learning_rate=self.config.sft_lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            save_strategy="epoch",
            logging_steps=10,
            bf16=True,
            deepspeed="configs/ds_zero3.json" if self.config.use_deepspeed else None,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        )
        trainer.train()
        trainer.save_model(self.config.sft_output_dir)
        logger.info("SFT complete → %s", self.config.sft_output_dir)


class PPOTrainer:
    """
    PPO training via OpenRLHF (§3.7 Stage 3).

    Wraps OpenRLHF's PPO infrastructure: actor (SFT checkpoint),
    critic (initialised from scratch), reference model (frozen SFT).
    """

    def __init__(self, config: TrainConfig):
        self.config = config

    def train(self, rl_examples_path: str) -> None:
        try:
            from openrlhf.trainer import PPOTrainer as ORLHFPPOTrainer
            from openrlhf.models import Actor, Critic
        except ImportError:
            logger.warning("OpenRLHF not installed; skipping PPO stage.")
            return

        import ray
        ray.init(ignore_reinit_error=True)

        logger.info("Starting PPO training with OpenRLHF")
        logger.info("  Actor LR        : %e", self.config.ppo_actor_lr)
        logger.info("  Critic LR       : %e", self.config.ppo_critic_lr)
        logger.info("  Clip eps        : %.2f", self.config.ppo_clip_eps)
        logger.info("  KL coeff        : %.3f", self.config.ppo_kl_coeff)
        logger.info("  Rollout batch   : %d",  self.config.ppo_rollout_batch)
        logger.info("  PPO iterations  : %d",  self.config.ppo_iterations)

        actor = Actor(self.config.sft_output_dir)
        critic = Critic(self.config.sft_output_dir)

        trainer = ORLHFPPOTrainer(
            actor=actor,
            critic=critic,
            actor_optim_config={"lr": self.config.ppo_actor_lr},
            critic_optim_config={"lr": self.config.ppo_critic_lr},
            clip_eps=self.config.ppo_clip_eps,
            kl_coeff=self.config.ppo_kl_coeff,
            entropy_coeff=self.config.ppo_entropy_coeff,
            value_coeff=self.config.ppo_value_coeff,
        )

        trainer.fit(
            data_path=rl_examples_path,
            rollout_batch_size=self.config.ppo_rollout_batch,
            train_batch_size=self.config.ppo_train_batch,
            ppo_epochs=self.config.ppo_epochs_per_batch,
            num_iterations=self.config.ppo_iterations,
            save_path=self.config.ppo_output_dir,
        )
        logger.info("PPO training complete → %s", self.config.ppo_output_dir)


# ===========================================================================
# End-to-end pipeline entry point
# ===========================================================================

class ESVRAGTrainer:
    """
    Orchestrates the full two-stage training pipeline.

    Usage
    -----
    trainer = ESVRAGTrainer(config)
    trainer.run(break_path="data/break/train.jsonl",
                monaco_path="data/monaco/train.json")
    """

    def __init__(self, config: Optional[ESVConfig] = None):
        self.config = config or ESVConfig()

    def run(self, break_path: Optional[str] = None,
             monaco_path: Optional[str] = None) -> None:

        train_cfg = self.config.train

        # ---------- Build planner for trace generation ----------
        generator = Generator(self.config.generator)
        retriever = RetrieverClient(self.config.retriever)
        planner   = MCTSPlanner(generator, retriever, self.config)

        # ---------- Stage 1: curate questions + generate traces ----------
        curator = DataCurator(train_cfg)
        questions = curator.curate(break_path, monaco_path)

        trace_gen = TraceGenerator(planner, train_cfg)
        sft_examples, rl_examples = trace_gen.generate_traces(
            questions, train_cfg.trace_output_dir
        )

        sft_path = str(Path(train_cfg.trace_output_dir) / "sft_examples.jsonl")
        rl_path  = str(Path(train_cfg.trace_output_dir) / "rl_examples.jsonl")

        # ---------- Stage 2a: SFT ----------
        sft_trainer = SFTTrainer(train_cfg)
        sft_trainer.train(sft_path)

        # ---------- Stage 2b: PPO ----------
        ppo_trainer = PPOTrainer(train_cfg)
        ppo_trainer.train(rl_path)

        logger.info("ESV-RAG training pipeline complete.")
