"""
ESV-RAG Evaluation Script

Evaluates ESV-RAG against 12 FlashRAG benchmark datasets using the
same metrics as Table 1 of the paper:
  - Exact Match (EM) for short-answer QA
  - Token F1 for multi-hop QA
  - Accuracy for multiple-choice questions

Usage:
    python evaluate.py --config configs/esv_mcts.yaml --dataset nq
    python evaluate.py --config configs/esv_mcts.yaml --all
"""

import argparse
import json
import logging
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from esv_rag.config import ESVConfig
from esv_rag.generator import Generator
from esv_rag.retriever import RetrieverClient
from esv_rag.planner import MCTSPlanner, SimplePlanner

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """Lower-case, strip punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall    = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def accuracy(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

FLASHRAG_DATASETS = {
    "nq":           ("nq", "em"),
    "trivia_qa":    ("trivia_qa", "em"),
    "hotpotqa":     ("hotpotqa", "f1"),
    "2wikimultihopqa": ("2wikimultihopqa", "f1"),
    "popqa":        ("popqa", "f1"),
    "webq":         ("webq", "em"),
    "musique":      ("musique", "f1"),
    "bamboogle":    ("bamboogle", "em"),
    "strategyqa":   ("strategyqa", "f1"),
    "asqa":         ("asqa", "f1"),
    "mmlu":         ("mmlu", "acc"),
    "arc_challenge":("arc_challenge", "acc"),
}


def load_dataset_examples(dataset_name: str, split: str = "test",
                           max_examples: Optional[int] = None) -> List[Dict]:
    """Load a FlashRAG benchmark via HuggingFace datasets."""
    try:
        from datasets import load_dataset as hf_load
        ds = hf_load(f"RUC-NLPIR/FlashRAG_datasets", dataset_name, split=split)
        examples = [dict(ex) for ex in ds]
        if max_examples:
            examples = examples[:max_examples]
        return examples
    except Exception as exc:
        logger.error("Could not load %s: %s", dataset_name, exc)
        return []


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate_dataset(planner,
                     examples: List[Dict],
                     metric_type: str,
                     dataset_name: str) -> Dict:
    scores       = []
    latencies    = []
    passages_cnt = []

    for i, ex in enumerate(examples):
        question = ex.get("question", ex.get("input", ""))
        gold     = ex.get("golden_answers", [ex.get("answer", "")])
        if isinstance(gold, str):
            gold = [gold]

        start  = time.time()
        result = planner.run(question)
        latency = time.time() - start
        latencies.append(latency)

        pred = result.final_state.main_answer if result.final_state else ""
        pred = pred or ""

        # Score against best-matching gold answer
        if metric_type == "em":
            score = max(exact_match(pred, g) for g in gold)
        elif metric_type == "f1":
            score = max(token_f1(pred, g) for g in gold)
        else:  # acc
            score = max(accuracy(pred, g) for g in gold)

        scores.append(score)
        passages_cnt.append(
            len(result.final_state.retrieved_documents) if result.final_state else 0
        )

        if (i + 1) % 50 == 0:
            logger.info("[%s] %d/%d — avg %s=%.3f",
                        dataset_name, i + 1, len(examples),
                        metric_type.upper(), sum(scores) / len(scores))

    return {
        "dataset":          dataset_name,
        "metric":           metric_type,
        "score":            round(sum(scores) / len(scores) * 100, 2) if scores else 0.0,
        "num_examples":     len(examples),
        "avg_latency":      round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "avg_passages":     round(sum(passages_cnt) / len(passages_cnt), 1) if passages_cnt else 0.0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_planner(config: ESVConfig, planner_type: str = "mcts"):
    generator = Generator(config.generator)
    retriever = RetrieverClient(config.retriever)

    if planner_type == "simple":
        from esv_rag.actions import (ExplorationAction, SolvingAction,
                                      VerificationAction)
        cfg = config.action
        explore = ExplorationAction(generator, retriever, cfg)
        solve   = SolvingAction(generator, retriever, cfg)
        verify  = VerificationAction(generator, retriever, cfg)
        return SimplePlanner(explore, solve, verify)

    return MCTSPlanner(generator, retriever, config)


def main():
    parser = argparse.ArgumentParser(description="Evaluate ESV-RAG on FlashRAG benchmarks")
    parser.add_argument("--config",  default="configs/esv_mcts.yaml",
                        help="Path to ESV-RAG YAML config")
    parser.add_argument("--dataset", default=None,
                        help="Single dataset to evaluate (e.g. nq, hotpotqa)")
    parser.add_argument("--all",     action="store_true",
                        help="Evaluate on all 12 FlashRAG benchmark datasets")
    parser.add_argument("--planner", default="mcts", choices=["mcts", "simple"],
                        help="Planner type (mcts=full model, simple=ablation)")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Cap number of examples per dataset (for quick testing)")
    parser.add_argument("--output",  default="results.json",
                        help="Output file for results")
    args = parser.parse_args()

    config = ESVConfig.from_yaml(args.config) if Path(args.config).exists() else ESVConfig()
    planner = build_planner(config, args.planner)

    datasets_to_eval = []
    if args.all:
        datasets_to_eval = list(FLASHRAG_DATASETS.keys())
    elif args.dataset:
        if args.dataset not in FLASHRAG_DATASETS:
            logger.error("Unknown dataset: %s", args.dataset)
            sys.exit(1)
        datasets_to_eval = [args.dataset]
    else:
        logger.error("Specify --dataset or --all")
        sys.exit(1)

    all_results = []
    for ds_name in datasets_to_eval:
        _, metric = FLASHRAG_DATASETS[ds_name]
        logger.info("=== Evaluating %s (%s) ===", ds_name, metric.upper())
        examples = load_dataset_examples(ds_name, max_examples=args.max_examples)
        if not examples:
            logger.warning("No examples loaded for %s; skipping", ds_name)
            continue
        res = evaluate_dataset(planner, examples, metric, ds_name)
        all_results.append(res)
        logger.info("  %s = %.2f%%", metric.upper(), res["score"])

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Dataset':<22} {'Metric':<6} {'Score':>7} {'Examples':>9} "
          f"{'Latency(s)':>11} {'Passages':>9}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['dataset']:<22} {r['metric']:<6} {r['score']:>7.2f} "
              f"{r['num_examples']:>9} {r['avg_latency']:>11.2f} "
              f"{r['avg_passages']:>9.1f}")
    print("=" * 70)

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
