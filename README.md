# EvoAgent — Self-Improving LLM Agent for Vietnamese Reading Comprehension

EvoAgent implements an iterative, LLM-driven prompt optimisation loop for the
[ViMMRC 2.0](https://huggingface.co/datasets/sonlam1102/vimmrc2.0) dataset.
At each iteration a **proposer** (Claude) designs a new prompting strategy, a
**Qwen2.5-7B-Instruct** model executes it, and a **reflector** (Claude)
analyses the failures to guide the next proposal.

This codebase is the starter code for an NLP course assignment. The teaching
team will wrap it into a Colab notebook; this repo is the pure-Python reference.

---

## Architecture overview

```
main.py              CLI entry point
harness.py           Orchestrates the T-iteration EvoAgent loop
├─ strategy.py       Strategy dataclass + StrategyHistory (disk persistence)
├─ model.py          QwenInference — batched inference, answer extraction
├─ executor.py       evaluate() — runs a strategy, returns EvalResult
├─ proposer.py       propose() — calls Claude to design a new strategy
├─ reflector.py      reflect() — calls Claude to analyse failures
└─ analysis.py       Learning curve, diversity, failure-mode plots
```

### EvoAgent loop (one iteration)

```
   ┌──────────────────────────────────────────────────────────────────┐
   │  1. propose(history) → Strategy          [Claude API call]       │
   │  2. evaluate(strategy, train_subset)     [Qwen inference]        │
   │  3. evaluate(strategy, dev)              [Qwen inference]        │
   │  4. reflect(strategy, dev_result)  → Reflection [Claude API]    │
   │  5. history.append_reflection(reflection)                        │
   │  6. Save everything to disk                                      │
   └──────────────────────────────────────────────────────────────────┘
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/your-org/evoagent.git
cd evoagent
pip install -r requirements.txt
```

### 2. Set your Anthropic API key

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

The proposer and reflector both use this key. Qwen inference is local and does
not require an API key.

### 3. (Optional) Pre-download the Qwen model

On Kaggle the first run will download ~15 GB. If you have internet access on
the notebook session this is automatic; otherwise pre-download with:

```bash
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
  AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct'); \
  AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct')"
```

---

## Running EvoAgent

### Basic run (5 iterations, default settings)

```bash
python main.py --T 5 --output-dir ./runs/exp01
```

### Verify the zero-shot baseline (iteration 0 only)

```bash
python main.py --T 1 --output-dir ./runs/baseline --dev-size 500
```

Expected dev accuracy: **~0.52**. This is the seed strategy: plain zero-shot
Qwen2.5-7B-Instruct with no CoT and no few-shot examples.

### Resume an interrupted run

```bash
python main.py --T 10 --output-dir ./runs/exp01 --resume ./runs/exp01/history.jsonl
```

### All options

```
--T INT                 Number of iterations (default: 5)
--resume PATH           Path to history.jsonl checkpoint
--dataset STR           HuggingFace dataset ID (default: sonlam1102/vimmrc2.0)
--train-size INT        Train-subset size for mid-loop eval (default: 100)
--dev-size INT          Dev split size; None = full split (default: None)
--seed INT              RNG seed for dataset shuffling (default: 42)
--model STR             Qwen model ID or local path (default: Qwen/Qwen2.5-7B-Instruct)
--batch-size INT        Inference batch size (default: 8; lower to 4 if OOM)
--max-new-tokens INT    Max generated tokens per example (default: 256)
--temperature FLOAT     Sampling temperature; 0.0 = greedy (default: 0.0)
--no-4bit               Disable 4-bit quantisation (needs ≥24 GB VRAM)
--claude-model STR      Claude model for propose/reflect (default: claude-sonnet-4-20250514)
--output-dir STR        Where to save results (default: ./runs/default)
--early-stop FLOAT      Stop when dev accuracy ≥ this value (default: 1.0)
--log-level STR         DEBUG / INFO / WARNING / ERROR (default: INFO)
--skip-analysis         Skip post-run plots
```

---

## Output structure

After a run with `--output-dir ./runs/exp01`:

```
runs/exp01/
├── args.json                          command-line args snapshot
├── run.log                            full timestamped log
├── history.jsonl                      resumable strategy history
├── iter_000_strategy.json             seed strategy
├── iter_000_eval_train.json           train-subset evaluation
├── iter_000_eval_dev.json             dev evaluation (with per-question records)
├── iter_001_strategy.json
├── iter_001_reflection.json
├── ...
├── learning_curve.pdf                 accuracy vs. iteration plot
├── strategy_diversity.pdf             pairwise prompt-template similarity heatmap
└── failure_mode_report.pdf / .txt     per-type accuracy trends + hypotheses
```

---

## Kaggle-specific notes

### Hardware target: T4 GPU (16 GB VRAM)

The default configuration uses 4-bit NF4 quantisation (`bitsandbytes`) which
keeps peak VRAM around **8–10 GB**. If you hit OOM errors:

1. Reduce `--batch-size` from 8 to 4.
2. Reduce `--max-new-tokens` from 256 to 128 (for non-CoT strategies).
3. If still OOM, add `--no-4bit` and expect ~14 GB VRAM (tight on T4).

### Expected runtimes on T4

| Component | Typical time |
|---|---|
| Model loading (4-bit) | 3–5 min |
| Evaluate 100 train examples | 2–4 min |
| Evaluate 500 dev examples | 8–15 min |
| Claude propose call | 5–15 sec |
| Claude reflect call | 10–20 sec |
| **One full iteration** | **~15–25 min** |
| **5-iteration run** | **~80–120 min** |

### Internet access on Kaggle

Kaggle notebooks require internet access to be enabled (Settings → Internet →
On) for the first model download and for Anthropic API calls. The model is
cached after the first download; subsequent runs in the same session are fast.

### Secrets on Kaggle

Store your Anthropic API key as a Kaggle secret named `ANTHROPIC_API_KEY`:

```python
from kaggle_secrets import UserSecretsClient
import os
os.environ["ANTHROPIC_API_KEY"] = UserSecretsClient().get_secret("ANTHROPIC_API_KEY")
```

---

## Module-level testing

Each component can be tested in isolation without loading the full Qwen model:

```python
# Test answer extraction
from model import extract_answer
assert extract_answer("Đáp án: B") == "B"
assert extract_answer("the answer is (C)") == "C"
assert extract_answer("A.") == "A"

# Test strategy serialisation
from strategy import make_seed_strategy
s = make_seed_strategy()
assert s == s.__class__.from_dict(s.to_dict())

# Test StrategyHistory persistence
import tempfile, pathlib
from strategy import StrategyHistory, make_seed_strategy
with tempfile.TemporaryDirectory() as d:
    h = StrategyHistory(pathlib.Path(d) / "history.jsonl")
    h.append_strategy(make_seed_strategy())
    h2 = StrategyHistory(pathlib.Path(d) / "history.jsonl")
    h2.load()
    assert len(h2) == 1
```

---

## Dataset: ViMMRC 2.0

- **Source:** `sonlam1102/vimmrc2.0` on HuggingFace
- **Task:** Multiple-choice reading comprehension from Vietnamese literature textbooks (grades 6–12)
- **Format:** Each example has a passage (100–400 words), a question, four answer choices (A/B/C/D), and a correct answer label.
- **Splits:** train / validation / test
- **Size:** ~2 000 train, ~400 validation, ~400 test examples

---

## Baseline

The seed strategy (iteration 0) is a zero-shot Vietnamese prompt:

```
Đọc đoạn văn sau và trả lời câu hỏi bằng cách chọn một trong các đáp án A, B, C hoặc D.

Đoạn văn:
{passage}

Câu hỏi: {question}

{choices}

Chỉ trả lời bằng một chữ cái duy nhất (A, B, C hoặc D).
```

Expected dev accuracy with Qwen2.5-7B-Instruct: **~52%**.

---

## Citation / acknowledgement

ViMMRC 2.0 dataset:
> Son T. Luu et al., "ViMMRC 2.0: Vietnamese Multiple-choice Machine Reading Comprehension", 2022.

Qwen2.5:
> Qwen Team, Alibaba Cloud. "Qwen2.5 Technical Report", 2024.
