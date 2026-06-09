"""
harness.py — Full EvoAgent training loop.

run_evoagent() orchestrates T iterations of:
  1. Propose a new strategy (or use the seed for iteration 0).
  2. Evaluate on the train subset.
  3. Evaluate on the dev split.
  4. Reflect on the results.
  5. Save state to disk.

State is saved after every step so interrupted runs can be resumed.
The harness tracks total token usage across both Qwen inference and meta-agent
calls.
"""

from __future__ import annotations

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
from self_proposer import propose_self
from self_reflector import reflect_self
from strategy import Strategy, StrategyHistory, StrategyMetadata, make_seed_strategy

logger = logging.getLogger(__name__)


@dataclass
class TokenBudget:
    """Running totals for token usage across the full run."""

    qwen_input: int = 0
    qwen_output: int = 0
    meta_total: int = 0

    def add_eval(self, result: EvalResult) -> None:
        self.qwen_input += result.total_input_tokens
        self.qwen_output += result.total_output_tokens

    def add_meta(self, tokens: int) -> None:
        self.meta_total += tokens

    def add_claude(self, tokens: int) -> None:
        """Backward-compatible alias for older call sites and histories."""
        self.add_meta(tokens)

    @property
    def qwen_total(self) -> int:
        return self.qwen_input + self.qwen_output

    def summary(self) -> str:
        return (
            f"Token usage — Qwen: {self.qwen_total:,} (in={self.qwen_input:,}, "
            f"out={self.qwen_output:,}) | Meta-agent: {self.meta_total:,}"
        )


def run_evoagent(
    T: int,
    train_dataset: Dataset,
    dev_dataset: Dataset,
    model: QwenInference,
    output_dir: Path,
    resume_from: Optional[Path] = None,
    early_stop_accuracy: float = 1.0,
    gemini_api_key: Optional[str] = None,
    gemini_model: str = "gemini-2.5-flash",
    self_optimize: bool = False,
) -> StrategyHistory:
    """
    Run the EvoAgent loop for up to T iterations.

    Parameters
    ----------
    T:
        Maximum number of iterations (including the seed strategy at iteration 0).
    train_dataset:
        Dataset used for cheap mid-loop evaluation (subset of train split).
    dev_dataset:
        Dataset used for the authoritative dev evaluation.
    model:
        A loaded QwenInference instance.
    output_dir:
        Directory where history, checkpoints, and per-iteration results are saved.
    resume_from:
        Path to an existing history.jsonl file to resume from.
    early_stop_accuracy:
        Stop the loop when dev accuracy reaches this threshold. Default 1.0
        (never stop early in practice).
    gemini_api_key:
        Google Gemini API key. If None, reads from GOOGLE_API_KEY env var.
    gemini_model:
        Which Gemini model to use for propose and reflect calls.

    Returns
    -------
    The populated StrategyHistory after all iterations.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history_path = output_dir / "history.jsonl"
    history = StrategyHistory(history_path)

    if resume_from is not None:
        # Load from an explicit checkpoint path.
        history.path = Path(resume_from)
        history.load()
        logger.info(
            "Resumed from %s — %d strategies already in history.",
            resume_from,
            len(history),
        )
    elif history_path.exists():
        # Auto-resume from the default path if it already exists.
        history.load()
        logger.info(
            "Auto-resumed from %s — %d strategies already in history.",
            history_path,
            len(history),
        )

    budget = TokenBudget()
    start_iteration = len(history.strategies)

    if start_iteration >= T:
        logger.info(
            "History already has %d strategies (T=%d). Nothing to do.",
            start_iteration,
            T,
        )
        return history

    logger.info(
        "Starting EvoAgent loop. Iterations %d–%d (T=%d).",
        start_iteration,
        T - 1,
        T,
    )

    for iteration in range(start_iteration, T):
        iter_start = time.time()
        logger.info("=" * 60)
        logger.info("ITERATION %d / %d", iteration, T - 1)
        logger.info("=" * 60)

        # ----------------------------------------------------------------
        # Step 1: Propose (or use seed for iteration 0)
        # ----------------------------------------------------------------
        meta_tokens_this_iter = 0

        if iteration == 0 and not history.strategies:
            strategy = make_seed_strategy()
            logger.info("Using seed strategy (iteration 0): id=%s.", strategy.id[:8])
        elif self_optimize:
            strategy, propose_tokens = propose_self(history, model=model)
            budget.add_meta(propose_tokens)
            meta_tokens_this_iter += propose_tokens
            logger.info(
                "Self-proposed strategy %s (iteration %d). Tokens: %d.",
                strategy.id[:8], iteration, propose_tokens,
            )
        else:
            strategy, propose_tokens = propose(
                history,
                api_key=gemini_api_key,
                model=gemini_model,
            )
            budget.add_meta(propose_tokens)
            meta_tokens_this_iter += propose_tokens
            logger.info(
                "Proposed strategy %s (iteration %d). Proposal tokens: %d.",
                strategy.id[:8], iteration, propose_tokens,
            )

        history.append_strategy(strategy)
        _save_strategy_json(strategy, output_dir, iteration)

        # ----------------------------------------------------------------
        # Step 2: Train-subset evaluation (cheap, ~50–100 examples)
        # ----------------------------------------------------------------
        logger.info("Running train-subset evaluation…")
        train_result = evaluate(
            strategy=strategy,
            split="train_subset",
            dataset=train_dataset,
            model=model,
        )
        budget.add_eval(train_result)
        _save_eval_result(train_result, output_dir, iteration, "train")
        logger.info(
            "Train-subset accuracy: %.3f (%d/%d) — %.1fs.",
            train_result.accuracy,
            train_result.num_correct,
            train_result.num_examples,
            train_result.elapsed_seconds,
        )

        # ----------------------------------------------------------------
        # Step 3: Dev evaluation (authoritative)
        # ----------------------------------------------------------------
        logger.info("Running dev evaluation…")
        dev_result = evaluate(
            strategy=strategy,
            split="dev",
            dataset=dev_dataset,
            model=model,
        )
        budget.add_eval(dev_result)
        _save_eval_result(dev_result, output_dir, iteration, "dev")
        logger.info(
            "Dev accuracy: %.3f (%d/%d) — %.1fs.",
            dev_result.accuracy,
            dev_result.num_correct,
            dev_result.num_examples,
            dev_result.elapsed_seconds,
        )

        # ----------------------------------------------------------------
        # Step 4: Update strategy metadata with scores
        # ----------------------------------------------------------------
        strategy.metadata.dev_accuracy = dev_result.accuracy
        strategy.metadata.train_accuracy = train_result.accuracy
        strategy.metadata.token_cost_qwen = (
            train_result.total_input_tokens
            + train_result.total_output_tokens
            + dev_result.total_input_tokens
            + dev_result.total_output_tokens
        )
        history.update_strategy_metadata(strategy.id, strategy.metadata)

        # ----------------------------------------------------------------
        # Step 5: Reflect (skip on last iteration to save API calls)
        # ----------------------------------------------------------------
        if iteration < T - 1:
            logger.info("Running reflection…")
            try:
                if self_optimize:
                    reflection, reflect_tokens = reflect_self(
                        strategy=strategy,
                        eval_result=dev_result,
                        model=model,
                    )
                else:
                    reflection, reflect_tokens = reflect(
                        strategy=strategy,
                        eval_result=dev_result,
                        api_key=gemini_api_key,
                        model=gemini_model,
                    )
                budget.add_meta(reflect_tokens)
                meta_tokens_this_iter += reflect_tokens
                strategy.metadata.token_cost_claude = meta_tokens_this_iter
                history.update_strategy_metadata(strategy.id, strategy.metadata)
                history.append_reflection(reflection)
                _save_reflection_json(reflection, output_dir, iteration)
                logger.info(
                    "Reflection stored. Reflect tokens: %d.", reflect_tokens
                )
            except Exception as exc:
                # Reflection failure should not abort the run.
                logger.error(
                    "Reflection failed for iteration %d: %s. Continuing without it.",
                    iteration,
                    exc,
                )

        # ----------------------------------------------------------------
        # Iteration summary
        # ----------------------------------------------------------------
        iter_elapsed = time.time() - iter_start
        logger.info(
            "Iteration %d complete in %.1fs. Dev=%.3f. %s",
            iteration,
            iter_elapsed,
            dev_result.accuracy,
            budget.summary(),
        )
        _print_leaderboard(history)

        # Early stopping
        if dev_result.accuracy >= early_stop_accuracy:
            logger.info(
                "Early stopping: dev accuracy %.3f >= threshold %.3f.",
                dev_result.accuracy,
                early_stop_accuracy,
            )
            break

    logger.info("EvoAgent loop finished. %s", budget.summary())
    best = history.best_strategy()
    if best is not None:
        logger.info(
            "Best strategy: id=%s, iteration=%d, dev_accuracy=%.3f.",
            best.id[:8],
            best.metadata.iteration,
            best.metadata.dev_accuracy,
        )

    return history


# ------------------------------------------------------------------
# Helpers
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
    import json

    path = output_dir / f"iter_{iteration:03d}_eval_{tag}.json"
    # Save the summary dict plus per-question records.
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
    import json

    path = output_dir / f"iter_{iteration:03d}_reflection.json"
    path.write_text(
        json.dumps(reflection.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.debug("Saved reflection to %s", path)


def _print_leaderboard(history: StrategyHistory) -> None:
    """Log a compact leaderboard of all strategies evaluated so far."""
    rows = history.summary_table()
    if not rows:
        return
    header = f"{'Iter':>4}  {'ID':>8}  {'CoT':>10}  {'Dev Acc':>8}  {'Train Acc':>9}  {'Meta tok':>10}  {'Qwen tok':>8}"
    logger.info("Leaderboard:\n%s", header)
    for r in rows:
        dev = f"{r['dev_accuracy']:.3f}" if r["dev_accuracy"] is not None else "  —  "
        train = f"{r['train_accuracy']:.3f}" if r["train_accuracy"] is not None else "  —  "
        logger.info(
            "  %4d  %8s  %10s  %8s  %9s  %10d  %8d",
            r["iteration"],
            r["id"],
            r["cot_format"],
            dev,
            train,
            r["meta_tokens"],
            r["qwen_tokens"],
        )
