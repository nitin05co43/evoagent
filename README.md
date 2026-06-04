# Assignment 3: EvoAgent — Self-Improving LLM Agent for Vietnamese Reading Comprehension

## Overview

In this assignment you will implement **EvoAgent**, an iterative self-improvement loop where a meta-agent LLM automatically proposes, evaluates, and refines prompting strategies for Vietnamese multiple-choice reading comprehension.

Instead of hand-crafting prompts, the agent learns from its own mistakes: after each evaluation it reflects on which question types it got wrong and proposes a better strategy for the next round.

**Dataset:** [ViMMRC 2.0](https://huggingface.co/datasets/uitnlp/vimmrc2.0) — Vietnamese literature reading comprehension, grades 6–12. Each passage has multiple questions with four answer choices (A/B/C/D).

**Baseline to beat:** 52% (majority class). Our reference implementation achieves **74.6%**.

---

## The EvoAgent Loop

```
Iteration 0: seed strategy (provided)
    ↓
┌─────────────────────────────────────────┐
│  1. Propose   — meta-agent generates    │
│               a new prompting strategy  │
│  2. Evaluate  — Qwen runs inference on  │
│               ViMMRC 2.0 train subset   │
│  3. Evaluate  — Qwen runs inference on  │
│               ViMMRC 2.0 dev split      │
│  4. Reflect   — meta-agent analyses     │
│               failures, forms hypothesis│
└──────── repeat for T iterations ────────┘
```

---

## What's Pre-Built (Do Not Modify)

| File | What it does |
|---|---|
| `model.py` | Loads Qwen2.5-7B with vLLM, runs batched inference |
| `executor.py` | Builds prompts, runs evaluation, computes accuracy by question type |
| `strategy.py` | Data structures: `Strategy`, `Reflection`, `StrategyHistory` |
| `analysis.py` | Learning curve plot, strategy diversity heatmap, failure mode report |
| `main.py` | CLI entry point — parses args, loads data, calls your harness |
| `run_modal.py` | Modal cloud GPU runner |

---

## What You Must Implement

### 1. `proposer.py` — Meta-agent strategy proposal

**`_build_system_prompt()`** — Write the system prompt that instructs the meta-agent to propose a new prompting strategy. The output must be valid JSON with fields: `hypothesis`, `prompt_template`, `cot_format`, `few_shot_examples`, `reasoning`.

**`_build_user_message(history)`** — Summarise the strategy history and latest reflection into a user message that tells the meta-agent what has been tried and what failed.

**`_parse_response(raw_text, iteration, parent_id)`** — Parse the JSON response into a `Strategy` object.

**`propose(history, ...)`** — Orchestrate the full call: build prompts → call LLM API → parse response → return `(Strategy, token_count)`. Include retry logic with exponential backoff.

### 2. `reflector.py` — Meta-agent failure analysis

**`_build_system_prompt()`** — Write the system prompt that instructs the meta-agent to diagnose why the current strategy failed. Hypotheses must be **specific and actionable**.

**`_build_user_message(strategy, eval_result)`** — Summarise the strategy and evaluation results (accuracy by type, top failure cases) for the meta-agent.

**`_parse_response(raw_text, strategy_id)`** — Parse the JSON response into a `Reflection` object.

**`reflect(strategy, eval_result, ...)`** — Orchestrate the full call and return `(Reflection, token_count)`.

### 3. `harness.py` — The EvoAgent loop

**`run_evoagent(T, train_dataset, dev_dataset, model, output_dir, ...)`** — Implement the full loop. The docstring contains a step-by-step implementation guide.

---

## Setup

### API Keys

You need a [Google Gemini API key](https://aistudio.google.com/app/apikey) (free tier works).

```bash
export GOOGLE_API_KEY=your_key_here
export HF_TOKEN=your_huggingface_token  # needed for ViMMRC 2.0
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run locally (CPU, small test)

```bash
python main.py \
    --T 2 \
    --train-size 20 \
    --dev-size 30 \
    --no-4bit \
    --output-dir ./runs/test
```

### Run on Modal (GPU, full experiment)

```bash
modal run --detach run_modal.py
```

---

## Deliverables

1. **Code** — your completed `proposer.py`, `reflector.py`, `harness.py`
2. **`history.jsonl`** — full iteration log (strategies + reflections + per-iteration accuracy)
3. **`learning_curve.pdf`** — accuracy vs. iteration plot (auto-generated)
4. **Report** (max 2 pages):
   - Best dev accuracy achieved and at which iteration
   - Which question types were hardest and why
   - Did the meta-agent's reflected strategies actually generalise to dev? Why or why not?

---

## Grading

| Component | Weight |
|---|---|
| Working end-to-end pipeline (5+ iterations) | 40% |
| Dev accuracy on ViMMRC 2.0 leaderboard | 30% |
| Report quality and analysis | 30% |

**Leaderboard baseline:** 52% (majority class) · **Reference:** 74.6%

---

## Tips

- Start with a simple meta-agent prompt and iterate — the same EvoAgent philosophy applies to building EvoAgent
- The `strategy.py` docstrings explain the data structures; read them before implementing
- `CoTFormat` has three options: `none`, `stepbystep`, `chain` — your proposer must output one of these
- A failed `reflect()` call should **not** crash the run — handle exceptions gracefully in the harness
- Use `modal run --detach` so the job survives network drops
