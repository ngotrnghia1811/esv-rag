# Data

## Evaluation Datasets (FlashRAG Benchmarks)

ESV-RAG is evaluated on 12 datasets from the [FlashRAG toolkit](https://github.com/RUC-NLPIR/FlashRAG).
They are loaded automatically from HuggingFace during evaluation:

| Dataset | Task Type | Metric | Test Size |
|---------|-----------|--------|-----------|
| NQ (Natural Questions) | Single-hop | EM | 3,610 |
| TriviaQA | Single-hop | EM | 11,313 |
| PopQA | Single-hop | F1 | 14,267 |
| WebQuestions | Single-hop | EM | 2,032 |
| HotpotQA | Multi-hop | F1 | 7,405 |
| 2WikiMultihopQA | Multi-hop | F1 | 12,576 |
| MuSiQue | Multi-hop | F1 | 2,417 |
| Bamboogle | Multi-hop | EM | 125 |
| StrategyQA | Complex | F1 | 2,290 |
| ASQA | Complex | F1 | 948 |
| MMLU | Multiple-choice | Acc | 14,042 |
| ARC-Challenge | Multiple-choice | Acc | 1,172 |

## Training Datasets

### BREAK

Complex question decomposition dataset (83,978 questions with QDMR annotations).

Download automatically:
```bash
python scripts/preprocess_data.py --break
```

This creates `data/break/train.jsonl`, `data/break/validation.jsonl`, `data/break/test.jsonl`.

Each line is a JSON object:
```json
{
    "question": "What is the capital of the country where ...",
    "answer": "Rome",
    "decomposition": ["Who painted the Mona Lisa?", "Where was ... born?", "What is the capital of Italy?"],
    "complexity": 0.5
}
```

### Monaco

Multi-document QA dataset (1,315 questions requiring synthesis across multiple documents).

Place manually at `data/monaco/train.json`:
```json
[
    {
        "question": "...",
        "answer": "...",
        "documents": ["doc1 text", "doc2 text", ...]
    }
]
```

## Retrieval Corpus

ESV-RAG uses the **DPR Wikipedia** corpus (Dec 2018, 21M passages) by default.

Download and index:
```bash
python scripts/preprocess_data.py --corpus wikipedia
```

For MS MARCO:
```bash
python scripts/preprocess_data.py --corpus msmarco
```

The corpus JSONL format (one document per line):
```json
{"id": "doc_001", "text": "Paris is the capital of France ..."}
```

Set the corpus path in `configs/esv_mcts.yaml`:
```yaml
retriever:
  corpus_path: "storage/corpus_cache/wikipedia.jsonl"
```
