# ESV-RAG

Official implementation of **"ESV-RAG: Explore-Solve-Verify Reasoning-Augmented Generation with Monte Carlo Tree Search Planning"**.

[[Paper]](docs/esv_rag_paper.pdf)

## Overview

ESV-RAG formulates multi-hop question answering as a **planning problem** solved by Monte Carlo Tree Search (MCTS). Instead of the conventional single-pass retrieve-then-generate pattern, ESV-RAG navigates a structured reasoning space through three coordinated actions:

- **(E) Explore** — generates diverse sub-queries using six reasoning skills grounded in cognitive science (Logical, Counterfactual, Probabilistic, Social, Contextual, Analogical).
- **(S) Solve** — performs targeted retrieval and two-phase answer synthesis: per-sub-question answering followed by main-answer integration.
- **(V) Verify** — adversarially self-evaluates answers through structured yes/no questions that detect logical inconsistencies and trigger iterative self-correction.

A learned policy (trained via PPO on MCTS-generated reasoning traces) selects optimal action sequences through PUCT-guided tree search, adapting to varying question complexity without relying on fixed retrieval patterns.

```
State:  s = (q, D, h, a_prev)
Actions: A = {Explore, Solve, Verify}
Reward:  r = 0.4·r_quality + 0.3·r_coherence + 0.2·r_verify + 0.1·r_efficiency
```

## Results

ESV-RAG achieves state-of-the-art performance across all 12 FlashRAG benchmark datasets, with the largest gains on complex multi-hop tasks:

| Method | NQ (EM) | TriviaQA (EM) | HotpotQA (F1) | 2Wiki (F1) | Musique (F1) | Bamboogle (EM) | Avg |
|--------|:-------:|:-------------:|:-------------:|:----------:|:------------:|:--------------:|:---:|
| Naive Generation | 22.6 | 55.7 | 28.4 | 33.9 | 24.1 | 8.0 | 28.8 |
| Standard RAG | 35.1 | 58.8 | 35.3 | 21.0 | 28.6 | 12.8 | 32.0 |
| IRCoT | 33.3 | 56.9 | 41.5 | 32.4 | 36.7 | 18.8 | 36.6 |
| Self-RAG | 36.4 | 38.2 | 29.6 | 25.1 | 28.9 | 16.4 | 29.1 |
| **ESV-RAG (Ours)** | **45.8** | **70.3** | **48.7** | **47.9** | **44.2** | **28.0** | **47.5** |
| vs. Best Baseline | +2.9 | +2.1 | +6.1 | +4.5 | +3.7 | +6.4 | — |

**Ablation Study (Table 2):**

| Variant | NQ (EM) | HotpotQA (F1) | 2Wiki (F1) | Musique (F1) | Avg |
|---------|:-------:|:-------------:|:----------:|:------------:|:---:|
| (1) No MCTS (fixed E→S→V) | 38.2 | 41.5 | 38.1 | 36.8 | 38.7 |
| (2) No Verification | 42.1 | 44.3 | 43.2 | 40.5 | 42.5 |
| (3) No Exploration | 35.8 | 36.9 | 35.4 | 33.2 | 35.3 |
| (4) Single skill only | 40.3 | 43.1 | 41.5 | 38.9 | 40.9 |
| **ESV-RAG (Full)** | **45.8** | **48.7** | **47.9** | **44.2** | **46.6** |

**Verification Effectiveness (Table 6):**

| Dataset | Before Verification | Corrections Triggered | After Verification | Gain |
|---------|:------------------:|:--------------------:|:-----------------:|:----:|
| NQ | 42.3% | 8.7% | 45.8% | +3.5% |
| HotpotQA | 44.1% | 12.4% | 48.7% | +4.6% |
| 2WikiMultihopQA | 43.2% | 13.9% | 47.9% | +4.7% |
| MuSiQue | 39.8% | 11.6% | 44.2% | +4.4% |

## Setup

```bash
conda create -n esv-rag python=3.10
conda activate esv-rag
pip install -r requirements.txt
```

**Environment variables:**
```bash
export OPENAI_API_KEY="your-key"     # if using OpenAI API
export HF_TOKEN="your-token"         # if using gated HuggingFace models
```

## Data

Download training and retrieval data:

```bash
# Training datasets (BREAK)
python scripts/preprocess_data.py --break

# Retrieval corpus (MS MARCO or Wikipedia)
python scripts/preprocess_data.py --corpus msmarco
```

See [data/README.md](data/README.md) for the full data guide.

## Evaluation

Evaluate ESV-RAG on a single dataset:
```bash
bash scripts/run_inference.sh configs/esv_mcts.yaml nq 500
```

Evaluate on all 12 FlashRAG benchmark datasets:
```bash
bash scripts/run_inference_all.sh configs/esv_mcts.yaml
```

Direct Python:
```bash
python evaluate.py --config configs/esv_mcts.yaml --dataset hotpotqa
python evaluate.py --config configs/esv_mcts.yaml --all --output results/full.json
```

Run the ablation baseline (fixed E→S→V, no MCTS):
```bash
python evaluate.py --config configs/esv_mcts.yaml --dataset nq --planner simple
```

## Training

Run the full two-stage training pipeline:

```bash
bash scripts/run_training.sh configs/training.yaml data/break/train.jsonl data/monaco/train.json
```

**Stage 1** — MCTS trace generation on 4,000 curated complex questions  
**Stage 2a** — Supervised fine-tuning on traces (Llama-3.2-1B, 4× H100, 4–6 h)  
**Stage 2b** — PPO training via OpenRLHF (500 iterations, 6–12 h)

Hardware requirements: 4× NVIDIA H100 80GB GPUs, Ray 2.8, DeepSpeed ZeRO-3.

## Programmatic Usage

```python
from esv_rag.config import ESVConfig
from esv_rag.generator import Generator
from esv_rag.retriever import RetrieverClient
from esv_rag.planner import MCTSPlanner

config    = ESVConfig.from_yaml("configs/esv_mcts.yaml")
generator = Generator(config.generator)
retriever = RetrieverClient(config.retriever)
planner   = MCTSPlanner(generator, retriever, config)

result = planner.run("What is the capital of the country where the Mona Lisa's painter was born?")
print(result.final_state.main_answer)   # "Rome"
print(result.action_sequence)           # ["E", "S", "V"]
print(f"Reward: {result.final_reward:.3f}")
```

## Architecture

```
esv_rag/
├── config.py     — ESVConfig, MCTSConfig, RewardConfig, ActionConfig, TrainConfig
├── state.py      — QuestionStatus, ReasoningSkill, QuestionItem, State (MDP state)
├── mcts.py       — Node, MCTS backbone, ESVMCTSNode, ESVMCTS (PUCT search)
├── actions.py    — ActionResult, ExplorationAction, SolvingAction, VerificationAction
├── generator.py  — Generator (OpenAI-compatible API / vLLM / HuggingFace)
├── retriever.py  — RetrieverClient (E5-base / BGE / BM25 + FAISS index)
├── rewards.py    — ESVRewardCalculator (4-component composite reward)
├── planner.py    — SimplePlanner (baseline), MCTSPlanner (full model)
└── train.py      — DataCurator, TraceGenerator, SFTTrainer, PPOTrainer
prompts/
├── explore.py    — EXPLORATION_PROMPT, FOCUSED_EXPLORATION_PROMPT,
│                   ITERATIVE_EXPLORATION_PROMPT
├── solve.py      — SUB_QUESTION_ANSWER_PROMPT, MAIN_ANSWER_SYNTHESIS_PROMPT,
│                   REANSWER_SUB_QUESTION_PROMPT
└── verify.py     — VERIFICATION_QUESTION_PROMPT, VERIFICATION_ANSWER_PROMPT,
                    SELF_CORRECTION_PROMPT, COMPREHENSIVE_VERIFICATION_PROMPT
```

**Key hyperparameters (§3.2, §3.7):**

| Parameter | Value | Description |
|-----------|:-----:|-------------|
| MCTS simulations | 50 | Rollouts per question |
| c_puct | 1.414 | PUCT exploration constant |
| γ (discount) | 0.95 | Reward discount factor |
| Reward weights (α₁…α₄) | 0.4 / 0.3 / 0.2 / 0.1 | Quality / Coherence / Verify / Efficiency |
| Verification threshold τ | 0.80 | Minimum v_score to pass |
| Max self-corrections | 2 | Correction iterations per episode |
| SFT base model | Llama-3.2-1B | Policy initialisation |
| PPO clip ε | 0.2 | Importance ratio clipping |
| KL coefficient | 0.01 | KL penalty against reference |

## Citation

```bibtex
@article{ngo2025esvrag,
    title={ESV-RAG: Explore-Solve-Verify Reasoning-Augmented Generation
           with Monte Carlo Tree Search Planning},
    author={Ngo, Nghia Trung and others},
    year={2025}
}
```

## License

MIT
