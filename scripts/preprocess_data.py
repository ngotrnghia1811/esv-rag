"""
Download and preprocess BREAK and Monaco datasets for ESV-RAG training.

BREAK (83,978 questions with QDMR decompositions):
    https://allenai.github.io/Break/

Monaco (1,315 multi-document QA questions):
    Provided as part of this repository (data/monaco/train.json)

Usage:
    python scripts/preprocess_data.py --break
    python scripts/preprocess_data.py --monaco
    python scripts/preprocess_data.py --all
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BREAK dataset
# ---------------------------------------------------------------------------

def download_break(output_dir: str = "data/break") -> None:
    """Download BREAK dataset from HuggingFace."""
    from datasets import load_dataset
    logger.info("Downloading BREAK dataset…")
    ds = load_dataset("break_data", "QDMR")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split in ["train", "validation", "test"]:
        split_path = out / f"{split}.jsonl"
        with open(split_path, "w") as f:
            for ex in ds[split]:
                obj = {
                    "question": ex.get("question_text", ""),
                    "answer":   ex.get("answer", ""),
                    "decomposition": ex.get("decomposition", "").split(";"),
                }
                obj["complexity"] = min(1.0, len(obj["decomposition"]) / 6.0)
                f.write(json.dumps(obj) + "\n")
        logger.info("Saved %s → %s", split, split_path)


def download_monaco(output_dir: str = "data/monaco") -> None:
    """
    Monaco dataset should be placed manually at data/monaco/train.json.
    Format: list of {"question": ..., "answer": ..., "documents": [...]}
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    placeholder = out / "README.md"
    placeholder.write_text(
        "# Monaco Dataset\n\n"
        "Place the Monaco multi-document QA dataset here as `train.json`.\n"
        "Format: [{\"question\": \"...\", \"answer\": \"...\", "
        "\"documents\": [\"doc1\", \"doc2\", ...]}, ...]\n"
    )
    logger.info("Created Monaco placeholder at %s", placeholder)


# ---------------------------------------------------------------------------
# Corpus preparation (MS MARCO / Wikipedia)
# ---------------------------------------------------------------------------

def prepare_corpus(corpus_name: str = "msmarco",
                   output_path: str = "storage/corpus_cache") -> None:
    """
    Download and convert a FlashRAG-supported corpus to JSONL.
    Supported: msmarco, wikipedia, hotpotqa
    """
    logger.info("Preparing corpus: %s", corpus_name)
    try:
        from datasets import load_dataset
        if corpus_name == "wikipedia":
            ds = load_dataset("wiki_dpr", "psgs_w100", split="train")
        elif corpus_name == "msmarco":
            ds = load_dataset("ms_marco", "v2.1", split="train")
        else:
            logger.error("Unknown corpus: %s", corpus_name)
            return

        out = Path(output_path) / f"{corpus_name}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for ex in ds:
                doc = {"id": ex.get("id", ""), "text": ex.get("passage", ex.get("text", ""))}
                f.write(json.dumps(doc) + "\n")
        logger.info("Corpus saved → %s", out)
    except Exception as exc:
        logger.error("Corpus preparation failed: %s", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess datasets for ESV-RAG")
    parser.add_argument("--break",   action="store_true", dest="do_break",
                        help="Download and preprocess BREAK dataset")
    parser.add_argument("--monaco",  action="store_true",
                        help="Prepare Monaco dataset directory")
    parser.add_argument("--corpus",  type=str, default=None,
                        help="Prepare retrieval corpus (msmarco | wikipedia)")
    parser.add_argument("--all",     action="store_true",
                        help="Run all preprocessing steps")
    args = parser.parse_args()

    if args.all or args.do_break:
        download_break()

    if args.all or args.monaco:
        download_monaco()

    if args.all or args.corpus:
        corpus = args.corpus or "msmarco"
        prepare_corpus(corpus)

    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
