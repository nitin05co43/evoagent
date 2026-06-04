"""
harness.py — EvoAgent training loop.

Your job: implement run_evoagent(), which orchestrates T iterations of:
  1. Propose a new strategy (or use the seed for iteration 0).
  2. Evaluate on the train subset (cheap sanity check).
  3. Evaluate on the dev split (authoritative score).
  4. Reflect on the results.
  5. Save state to disk.

The seed strategy (iteration 0) is pre-built — import make_seed_strategy().
State is saved after every step so interrupted runs can be resumed.

TODO: Implement run_evoagent() below.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from datasets import Dataset

from executor import EvalResult, evaluate
from model import QwenInference
from proposer import propose
from reflector import reflect
from strategy import Strategy, StrategyHistory, StrategyMetadata, make_seed_strategy

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Token budget tracker (provided — no changes needed)
# ------------------------------------------------------------------

@dataclass
class TokenBudget:
    """Running totals for token usage across the full run."""

    qwen_input: int = 0
    qwen_output: int = 0
    meta_agent_total: int = 0

    def add_eval(self, result: EvalResult) -> None:
        self.qwen_input += result.total_input_tokens
        self.qwen_output += result.total_output_tokens

    def add_meta_agent(self, tokens: int) -> None:
        self.meta_agent_total += tokens

    @property
    def qwen_total(self) -> int:
        return self.qwen_input + self.qwen_output

    def summary(self) -> str:
        return (
            f"Token usage — Qwen: {self.qwen_total:,} "
            f"(in={self.qwen_input:,}, out={self.qwen_output:,}) "
            f"| Meta-agent: {self.meta_agent_total:,}"
        )


# ------------------------------------------------------------------
# TODO — Implement the EvoAgent loop
# ------------------------------------------------------------------

def run_evoagent(
    T: int,
    train_dataset: Dataset,
    dev_dataset: Dataset,
    model: QwenInference,
    output_dir: Path,
    resume_from: Optional[Path] = None,
    early_stop_accuracy: float = 1.0,
    gemini_api_key: Optional[str] = None,
    gemini_model: str = "gemini-2.0-flash",
) -> StrategyHistory:
    """
    Run the EvoAgent loop for up to T iterations.

    Parameters
    ----------
    T:
        Total number of iterations (iteration 0 = seed strategy).
    train_dataset:
        Small subset of the training split for cheap mid-loop eval.
    dev_dataset:
        Full dev split for authoritative accuracy numbers.
    model:
        A loaded QwenInference instance (call model.generate_batch()).
    output_dir:
        Directory where history, checkpoints, and per-iteration results are saved.
    resume_from:
        Path to an existing history.jsonl file to resume an interrupted run.
    early_stop_accuracy:
        Stop early if dev accuracy reaches this threshold (default 1.0 = never).
    gemini_api_key:
        Meta-agent API key. If None, reads from GOOGLE_API_KEY env var.
    gemini_model:
        Meta-agent model name for propose() and reflect() calls.

    Returns
    -------
    The populated StrategyHistory after all iterations.

    Implementation guide
    --------------------
    1. Create output_dir and initialise a StrategyHistory(history_path).

    2. Resume: if resume_from is given, call history.load() on that path.
       If history_path already exists, auto-resume from it.
       Set start_iteration = len(history.strategies).
       If start_iteration >= T, log "Nothing to do" and return early.

    3. Loop for iteration in range(start_iteration, T):

       a. PROPOSE
          - iteration == 0 and history is empty → use make_seed_strategy()
          - otherwise → call propose(history, api_key=..., model=...)
          - call history.append_strategy(strategy)
          - save strategy to disk with _save_strategy_json()

       b. EVALUATE on train subset
          - call evaluate(strategy, "train_subset", train_dataset, model)
          - save result with _save_eval_result(..., tag="train")

       c. EVALUATE on dev split
          - call evaluate(strategy, "dev", dev_dataset, model)
          - save result with _save_eval_result(..., tag="dev")

       d. UPDATE strategy metadata
          - strategy.metadata.dev_accuracy = dev_result.accuracy
          - strategy.metadata.train_accuracy = train_result.accuracy
          - call history.update_strategy_metadata(strategy.id, strategy.metadata)

       e. REFLECT (skip on the last iteration to save API calls)
          - call reflect(strategy, dev_result, api_key=..., model=...)
          - call history.append_reflection(reflection)
          - save reflection with _save_reflection_json()
          - wrap in try/except — a failed reflection should not abort the run

       f. Log a leaderboard with _print_leaderboard(history)

       g. Check early stopping

    4. Return history.
    """
    # TODO: Implement the EvoAgent loop.
    raise NotImplementedError


# ------------------------------------------------------------------
# File I/O helpers (provided — no changes needed)
# ------------------------------------------------------------------

def _save_strategy_json(strategy: Strategy, output_dir: Path, iteration: int) -> None:
    path = output_dir / f"iter_{iteration:03d}_strategy.json"
    path.write_text(strategy.to_json(), encoding="utf-8")
    logger.debug("Saved strategy to %s", path)


def _save_eval_result(
    result: EvalResult,
    output_dir: Path,
    iteration: int,
    tag: str,
) -> None:
    path = output_dir / f"iter_{iteration:03d}_eval_{tag}.json"
    data = result.to_dict()
    data["per_question"] = [
        {
            "question_id": r.question_id,
            "question": r.question,
            "gold_answer": r.gold_answer,
            "predicted_answer": r.predicted_answer,
            "is_correct": r.is_correct,
            "question_type": r.question_type,
            "raw_output": r.raw_output,
        }
        for r in result.per_question
    ]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.debug("Saved %s eval result to %s", tag, path)


def _save_reflection_json(reflection, output_dir: Path, iteration: int) -> None:
    path = output_dir / f"iter_{iteration:03d}_reflection.json"
    path.write_text(
        json.dumps(reflection.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.debug("Saved reflection to %s", path)


def _print_leaderboard(history: StrategyHistory) -> None:
    rows = history.summary_table()
    if not rows:
        return
    header = (
        f"{'Iter':>4}  {'ID':>8}  {'CoT':>10}  "
        f"{'Dev Acc':>8}  {'Train Acc':>9}  {'Meta tok':>10}  {'Qwen tok':>8}"
    )
    logger.info("Leaderboard:\n%s", header)
    for r in rows:
        dev = f"{r['dev_accuracy']:.3f}" if r["dev_accuracy"] is not None else "  —  "
        train = f"{r['train_accuracy']:.3f}" if r["train_accuracy"] is not None else "  —  "
        logger.info(
            "  %4d  %8s  %10s  %8s  %9s  %10d  %8d",
            r["iteration"], r["id"], r["cot_format"],
            dev, train, r["claude_tokens"], r["qwen_tokens"],
        )
