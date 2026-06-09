# EvoAgent — Self-Improving LLM Agent for Vietnamese Reading Comprehension

EvoAgent implements an iterative, LLM-driven prompt optimisation loop for the
[ViMMRC 2.0](https://huggingface.co/datasets/uitnlp/vimmrc2.0) dataset.
At each iteration a **meta-agent** designs a new prompting strategy,
**Qwen2.5-7B-Instruct-AWQ** executes it via vLLM, and the meta-agent analyses
failures to guide the next proposal.

Two modes are supported:
- **Gemini-guided** (default): Gemini 2.5-flash acts as meta-agent
- **Self-optimization** (`--self-optimize`): Qwen guides itself — no external API needed

Reference results on ViMMRC 2.0 dev split:
- Gemini-guided: **74.6%** (best at iteration 0)
- Self-guided: **74.6%** (best at iteration 0)

---

## Architecture overview

```
main.py              CLI entry point
harness.py           Orchestrates the T-iteration EvoAgent loop
├─ strategy.py       Strategy dataclass + StrategyHistory (disk persistence)
├─ model.py          QwenInference — vLLM batched inference + answer extraction
├─ executor.py       evaluate() — runs a strategy, returns EvalResult
├─ proposer.py       propose() — calls Gemini to design a new strategy
├─ reflector.py      reflect() — calls Gemini to analyse failures
├─ self_proposer.py  propose_self() — Qwen proposes its own strategy
├─ self_reflector.py reflect_self() — Qwen analyses its own failures
├─ data.py           Dataset loader (HuggingFace or local)
└─ analysis.py       Learning curve, diversity, failure-mode plots
```

### EvoAgent loop (one iteration)

```
   ┌──────────────────────────────────────────────────────────────────┐
   │  1. propose(history) → Strategy     [meta-agent call]            │
   │  2. evaluate(strategy, train_subset) [Qwen vLLM inference]       │
   │  3. evaluate(strategy, dev)          [Qwen vLLM inference]       │
   │  4. reflect(strategy, dev_result) → Reflection [meta-agent call] │
   │  5. history.append_reflection(reflection)                        │
   │  6. Save everything to disk                                      │
   └──────────────────────────────────────────────────────────────────┘
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/nitin05co43/evoagent.git
cd evoagent
pip install -r requirements.txt
```

### 2. API keys

**Gemini-guided mode** — get a free key at https://aistudio.google.com/apikey:
```bash
export GOOGLE_API_KEY="your-key-here"
```

**Self-optimization mode** — no API key needed.

**HuggingFace token** (required for the dataset):
1. Accept dataset terms at https://huggingface.co/datasets/uitnlp/vimmrc2.0
2. Get a read token at https://huggingface.co/settings/tokens

```bash
export HF_TOKEN="hf_..."
```

---

## Running on Modal (recommended)

Modal gives you a T4 GPU in the cloud with no local setup required.

### 1. Install Modal and log in

```bash
pip install modal
modal setup
```

### 2. Store your secrets

```bash
modal secret create google GOOGLE_API_KEY=your-gemini-key
modal secret create huggingface HF_TOKEN=hf_your-token
```

### 3. Run

Gemini-guided (default):
```bash
modal run --detach run_modal.py
```

Self-optimization (Qwen guides itself):
```bash
modal run --detach run_modal.py::self
```

Results are saved to a Modal volume and can be downloaded:

```bash
modal volume get evoagent-runs exp01 ./results       # Gemini-guided
modal volume get evoagent-runs exp_self ./results    # Self-guided
```

---

## Running locally

```bash
python main.py --T 5 --output-dir ./runs/exp01
```

Self-optimization mode:
```bash
python main.py --T 5 --output-dir ./runs/exp_self --self-optimize
```

### All CLI options

```
--T INT                 Number of iterations (default: 5)
--resume PATH           Path to history.jsonl checkpoint
--dataset STR           HuggingFace dataset ID (default: uitnlp/vimmrc2.0)
--train-size INT        Train-subset size for mid-loop eval (default: 100)
--dev-size INT          Dev split size; None = full split (default: None)
--seed INT              RNG seed for dataset shuffling (default: 42)
--model STR             Qwen model ID or local path (default: Qwen/Qwen2.5-7B-Instruct-AWQ)
--batch-size INT        Inference batch size (default: 8; lower to 4 if OOM)
--max-new-tokens INT    Max generated tokens per example (default: 256)
--temperature FLOAT     Sampling temperature; 0.0 = greedy (default: 0.0)
--no-4bit               Disable 4-bit quantisation (needs ≥24 GB VRAM)
--gemini-model STR      Gemini model for propose/reflect (default: gemini-2.5-flash)
--self-optimize         Use Qwen as its own meta-agent — no Gemini API needed
--output-dir STR        Where to save results (default: ./runs/default)
--early-stop FLOAT      Stop when dev accuracy ≥ this value (default: 1.0)
--log-level STR         DEBUG / INFO / WARNING / ERROR (default: INFO)
--skip-analysis         Skip post-run plots
```

---

## Output structure

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

## Hardware requirements

### Modal / cloud (recommended)
- T4 GPU (16 GB VRAM) — default in `run_modal.py`
- AWQ quantisation keeps peak VRAM at ~6–8 GB

### Local
- NVIDIA GPU with ≥16 GB VRAM
- If you hit OOM: reduce `--batch-size` to 4, or reduce `--max-new-tokens` to 128

### Expected runtimes on T4

| Component | Typical time |
|---|---|
| Model loading (AWQ) | 3–5 min |
| Evaluate 100 train examples | ~10 min |
| Evaluate full dev split (~560 examples) | ~15 min |
| Gemini propose + reflect | ~30 sec |
| Self-propose + reflect (Qwen) | ~2 min |
| **One full iteration** | **~30 min** |
| **5-iteration run** | **~2.5–3 hours** |

### Cost estimate (Modal + Gemini)

| Component | Cost per iteration |
|---|---|
| Modal T4 GPU (~30 min) | ~$0.18 |
| Gemini 2.5-flash (propose + reflect) | ~$0.02 |
| **Total per iteration** | **~$0.20** |
| **5-iteration run** | **~$1.00** |

Self-optimization mode has no Gemini cost.

---

## Dataset: ViMMRC 2.0

- **Source:** `uitnlp/vimmrc2.0` on HuggingFace (gated — requires free account + terms agreement)
- **Task:** Multiple-choice reading comprehension from Vietnamese literature textbooks (grades 6–12)
- **Format:** passage (100–400 words) + question + four choices (A/B/C/D) + correct answer
- **Splits:** train / validation / test

---

## Seed strategy (iteration 0 baseline)

```
Đọc đoạn văn sau và trả lời câu hỏi bằng cách chọn một trong các đáp án A, B, C hoặc D.

Đoạn văn:
{passage}

Câu hỏi: {question}

{choices}

Chỉ trả lời bằng một chữ cái duy nhất (A, B, C hoặc D).
```

Dev accuracy with Qwen2.5-7B-Instruct-AWQ: **74.6%**. Zero-shot, no CoT, no few-shot examples. This was the best-performing strategy across all 5 iterations in both modes — adding CoT consistently hurt performance.

---

## Citation

ViMMRC 2.0:
> Son T. Luu et al., "ViMMRC 2.0: Vietnamese Multiple-choice Machine Reading Comprehension", 2022.

Qwen2.5:
> Qwen Team, Alibaba Cloud. "Qwen2.5 Technical Report", 2024.


---

## Architecture overview

```
main.py              CLI entry point
harness.py           Orchestrates the T-iteration EvoAgent loop
├─ strategy.py       Strategy dataclass + StrategyHistory (disk persistence)
├─ model.py          QwenInference — batched inference, answer extraction
├─ executor.py       evaluate() — runs a strategy, returns EvalResult
├─ proposer.py       propose() — calls Gemini to design a new strategy
├─ reflector.py      reflect() — calls Gemini to analyse failures
├─ data.py           Dataset loader (HuggingFace or local)
└─ analysis.py       Learning curve, diversity, failure-mode plots
```

### EvoAgent loop (one iteration)

```
   ┌──────────────────────────────────────────────────────────────────┐
   │  1. propose(history) → Strategy          [Gemini API call]       │
   │  2. evaluate(strategy, train_subset)     [Qwen inference]        │
   │  3. evaluate(strategy, dev)              [Qwen inference]        │
   │  4. reflect(strategy, dev_result) → Reflection  [Gemini API]    │
   │  5. history.append_reflection(reflection)                        │
   │  6. Save everything to disk                                      │
   └──────────────────────────────────────────────────────────────────┘
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/nitin05co43/evoagent.git
cd evoagent
git checkout claude/elegant-ptolemy-pl6HE
pip install -r requirements.txt
```

### 2. Get a Google Gemini API key

Go to **https://aistudio.google.com/apikey** → Create API key → copy it.

```bash
export GOOGLE_API_KEY="your-key-here"
```

The proposer and reflector both use this key. Qwen inference is local.

### 3. Get a HuggingFace token (required for the dataset)

1. Create a free account at **https://huggingface.co/join**
2. Accept dataset terms at **https://huggingface.co/datasets/uitnlp/vimmrc2.0**
3. Get a read token at **https://huggingface.co/settings/tokens**

```bash
export HF_TOKEN="hf_..."
```

---

## Running on Modal (recommended)

Modal gives you a T4 GPU in the cloud with no local setup required.

### 1. Install Modal and log in

```bash
pip install modal
modal setup
```

### 2. Store your secrets

```bash
modal secret create google GOOGLE_API_KEY=your-gemini-key
modal secret create huggingface HF_TOKEN=hf_your-token
```

### 3. Run

```bash
modal run run_modal.py
```

Logs stream live. Results are saved to a Modal volume and can be downloaded:

```bash
modal volume get evoagent-runs exp01 ./results
```

---

## Running locally

```bash
python main.py --T 5 --output-dir ./runs/exp01
```

### Verify the zero-shot baseline (iteration 0 only)

```bash
python main.py --T 1 --output-dir ./runs/baseline --dev-size 200
```

Expected dev accuracy: **~52%**. This is the seed strategy: plain zero-shot
Qwen2.5-7B-Instruct with no CoT and no few-shot examples.

### Resume an interrupted run

```bash
python main.py --T 10 --output-dir ./runs/exp01 --resume ./runs/exp01/history.jsonl
```

### All CLI options

```
--T INT                 Number of iterations (default: 5)
--resume PATH           Path to history.jsonl checkpoint
--dataset STR           HuggingFace dataset ID (default: uitnlp/vimmrc2.0)
--train-size INT        Train-subset size for mid-loop eval (default: 100)
--dev-size INT          Dev split size; None = full split (default: None)
--seed INT              RNG seed for dataset shuffling (default: 42)
--model STR             Qwen model ID or local path (default: Qwen/Qwen2.5-7B-Instruct)
--batch-size INT        Inference batch size (default: 8; lower to 4 if OOM)
--max-new-tokens INT    Max generated tokens per example (default: 256)
--temperature FLOAT     Sampling temperature; 0.0 = greedy (default: 0.0)
--no-4bit               Disable 4-bit quantisation (needs ≥24 GB VRAM)
--gemini-model STR      Gemini model for propose/reflect (default: gemini-2.5-flash)
--output-dir STR        Where to save results (default: ./runs/default)
--early-stop FLOAT      Stop when dev accuracy ≥ this value (default: 1.0)
--log-level STR         DEBUG / INFO / WARNING / ERROR (default: INFO)
--skip-analysis         Skip post-run plots
```

---

## Output structure

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

## Hardware requirements

### Modal / cloud (recommended)
- T4 GPU (16 GB VRAM) — default in `run_modal.py`
- 4-bit NF4 quantisation keeps peak VRAM at ~8–10 GB

### Local
- NVIDIA GPU with ≥16 GB VRAM
- If you hit OOM: reduce `--batch-size` to 4, or reduce `--max-new-tokens` to 128

### Expected runtimes on T4

| Component | Typical time |
|---|---|
| Model loading (4-bit) | 3–5 min |
| Evaluate 100 train examples | ~20 min |
| Evaluate full dev split (~400 examples) | ~60 min |
| Gemini propose call | 5–20 sec |
| Gemini reflect call | 10–30 sec |
| **One full iteration** | **~80–90 min** |
| **5-iteration run** | **~7–8 hours** |

> **Tip:** Use `--dev-size 100` for faster iteration during development.

### Cost estimate (Modal + Gemini)

| Component | Cost per iteration |
|---|---|
| Modal T4 GPU (~90 min) | ~$0.54 |
| Gemini 2.5-flash (propose + reflect) | ~$0.02 |
| **Total per iteration** | **~$0.56** |
| **5-iteration run** | **~$2.80** |

---

## Module-level testing

Each component can be tested without loading the full Qwen model:

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

- **Source:** `uitnlp/vimmrc2.0` on HuggingFace (gated — requires free account + terms agreement)
- **Task:** Multiple-choice reading comprehension from Vietnamese literature textbooks (grades 6–12)
- **Format:** passage (100–400 words) + question + four choices (A/B/C/D) + correct answer
- **Splits:** train / validation / test (~2,000 / ~400 / ~400 examples)

---

## Baseline

The seed strategy (iteration 0) is a plain zero-shot Vietnamese prompt with no
chain-of-thought and no few-shot examples:

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

## Citation

ViMMRC 2.0:
> Son T. Luu et al., "ViMMRC 2.0: Vietnamese Multiple-choice Machine Reading Comprehension", 2022.

Qwen2.5:
> Qwen Team, Alibaba Cloud. "Qwen2.5 Technical Report", 2024.
