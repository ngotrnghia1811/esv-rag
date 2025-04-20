"""
Configuration dataclasses for ESV-RAG.

Covers all subsystems: MCTS tree search, reward weighting, ESV action parameters,
retrieval/generation backends, and PPO training.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MCTSConfig:
    """Monte Carlo Tree Search hyperparameters."""
    num_simulations: int = 50
    max_depth: int = 10
    exploration_constant: float = 1.414   # c_puct (sqrt(2))
    discount_factor: float = 0.95
    max_simulation_steps: int = 10        # steps per rollout during simulation
    verbose: bool = False


@dataclass
class RewardConfig:
    """Weights for the composite reward function (must sum to 1.0)."""
    answer_quality: float = 0.4
    reasoning_coherence: float = 0.3
    verification_success: float = 0.2
    efficiency: float = 0.1
    max_expected_steps: int = 15
    optimal_questions: int = 5
    total_skills: int = 6

    def __post_init__(self):
        total = (self.answer_quality + self.reasoning_coherence +
                 self.verification_success + self.efficiency)
        assert abs(total - 1.0) < 1e-3, f"Reward weights must sum to 1.0, got {total}"


@dataclass
class ActionConfig:
    """Per-action generation and search settings."""
    # Exploration
    max_explore_questions: int = 5
    explore_temperature: float = 0.8
    # Solving
    max_solve_workers: int = 4
    solve_temperature: float = 0.7
    confidence_threshold: float = 0.7
    enable_parallel_solve: bool = True
    # Verification
    max_verify_questions: int = 5
    verify_temperature: float = 0.6
    verification_threshold: float = 0.80
    enable_self_correction: bool = True
    max_correction_attempts: int = 2


@dataclass
class GeneratorConfig:
    """LLM generation backend settings."""
    # Online (OpenAI-compatible API)
    api_url: Optional[str] = None
    api_key: str = "EMPTY"
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    max_tokens: int = 1024
    temperature: float = 0.8
    top_p: float = 0.9
    # Offline (vLLM / HuggingFace)
    use_vllm: bool = False
    gpu_memory_utilization: float = 0.85
    # Caching
    use_cache: bool = True
    cache_dir: str = "cache/generator"


@dataclass
class RetrieverConfig:
    """Retrieval backend settings (FlashRAG / BM25 / dense)."""
    method: str = "e5base"           # "bm25" | "e5base" | "bge"
    corpus_path: str = ""
    index_path: str = "storage/indexes"
    retrieval_topk: int = 5
    query_max_length: int = 512
    use_cache: bool = True
    cache_path: str = "cache/retrieval"
    model_path: str = "intfloat/e5-base-v2"
    batch_size: int = 32


@dataclass
class TrainConfig:
    """Two-stage training pipeline settings."""
    # Stage 1 — MCTS trace generation
    num_curated_questions: int = 4000
    complexity_threshold: float = 0.6
    trace_output_dir: str = "train_data/traces"
    data_multiply_factor: int = 6     # SFT + RL examples per trace

    # Stage 2a — Supervised fine-tuning
    sft_base_model: str = "meta-llama/Llama-3.2-1B"
    sft_lr: float = 5e-5
    sft_batch_size: int = 128
    sft_epochs: int = 3
    sft_output_dir: str = "checkpoints/sft"

    # Stage 2b — PPO with OpenRLHF
    ppo_actor_lr: float = 5e-7
    ppo_critic_lr: float = 9e-6
    ppo_clip_eps: float = 0.2
    ppo_kl_coeff: float = 0.01
    ppo_entropy_coeff: float = 0.001
    ppo_value_coeff: float = 0.5
    ppo_rollout_batch: int = 512
    ppo_train_batch: int = 64
    ppo_epochs_per_batch: int = 4
    ppo_iterations: int = 500
    ppo_output_dir: str = "checkpoints/ppo"

    # Infrastructure
    num_gpus: int = 4
    use_deepspeed: bool = True
    deepspeed_stage: int = 3


@dataclass
class ESVConfig:
    """Top-level configuration composing all subsystem configs."""
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # Evaluation datasets (FlashRAG benchmarks)
    eval_datasets: List[str] = field(default_factory=lambda: [
        "nq", "trivia_qa", "hotpotqa", "2wikimultihopqa",
        "popqa", "webq", "musique", "bamboogle",
        "strategyqa", "asqa", "mmlu", "arc_challenge",
    ])

    @classmethod
    def from_yaml(cls, path: str) -> "ESVConfig":
        import yaml
        from dataclasses import fields as dc_fields

        with open(path) as f:
            raw = yaml.safe_load(f)

        def _fill(dc_cls, d: dict):
            valid = {f.name for f in dc_fields(dc_cls)}
            return dc_cls(**{k: v for k, v in d.items() if k in valid})

        cfg = cls()
        if "mcts" in raw:
            cfg.mcts = _fill(MCTSConfig, raw["mcts"])
        if "reward" in raw:
            cfg.reward = _fill(RewardConfig, raw["reward"])
        if "action" in raw:
            cfg.action = _fill(ActionConfig, raw["action"])
        if "generator" in raw:
            cfg.generator = _fill(GeneratorConfig, raw["generator"])
        if "retriever" in raw:
            cfg.retriever = _fill(RetrieverConfig, raw["retriever"])
        if "train" in raw:
            cfg.train = _fill(TrainConfig, raw["train"])
        return cfg
