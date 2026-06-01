"""
reflector.py — LLM-driven reflection on strategy performance.

After each evaluation the reflector calls Claude with the eval results to
produce a structured Reflection. This reflection is stored in the history and
passed to the next propose() call so the proposer can build on concrete
evidence rather than guessing.

The reflection includes:
  - Accuracy by question type (computed from per-question results)
  - Top-5 failure cases with brief analysis
  - A concrete, falsifiable hypothesis for the next iteration
  - A prose summary
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Optional

import anthropic
from pydantic import BaseModel, Field

from executor import EvalResult, QuestionResult
from strategy import Reflection, Strategy

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Pydantic schema for Claude's structured reflection output
# ------------------------------------------------------------------


class FailureAnalysis(BaseModel):
    """Analysis of a single failure case."""

    question: str = Field(description="The question text (truncated if long).")
    gold_answer: str = Field(description="The correct answer letter.")
    predicted_answer: Optional[str] = Field(description="What the model predicted.")
    raw_output_excerpt: str = Field(description="First 200 chars of model output.")
    analysis: str = Field(
        description="Why did the model get this wrong? Be specific."
    )


class StructuredReflection(BaseModel):
    """The full structured reflection returned by Claude."""

    accuracy_by_type: dict[str, float] = Field(
        description=(
            "Accuracy for each question type as a float 0–1. "
            "Keys must match the types provided."
        )
    )
    weakest_type: str = Field(
        description="The question type with the lowest accuracy."
    )
    top_failures: list[FailureAnalysis] = Field(
        description="Analysis of the 5 most informative failure cases.",
        max_length=5,
    )
    hypothesis: str = Field(
        description=(
            "A specific, testable hypothesis about why the strategy fails "
            "on the weakest type and what a better strategy should do. "
            "Must be actionable: a proposer should be able to directly "
            "derive a new prompt template from this hypothesis."
        )
    )
    summary: str = Field(
        description="One paragraph summarising the reflection and its implications."
    )


# ------------------------------------------------------------------
# System and user prompt builders
# ------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an NLP research assistant analysing the performance of a prompting
strategy on Vietnamese multiple-choice reading comprehension (ViMMRC 2.0).

You will receive:
  1. A summary of the strategy (prompt template, CoT format, few-shot examples).
  2. Overall accuracy and per-category accuracy.
  3. The 10 worst failure cases with the model's raw output.

Your job is to produce a structured reflection that will guide the next
strategy proposal. Be concrete and diagnostic, not generic.

Bad hypothesis: "The model needs to understand the passage better."
Good hypothesis: "The model fails on 'cause_effect' questions because the
  zero-shot prompt does not ask it to explicitly identify causal chains.
  A prompt that asks 'What caused X according to the passage?' before
  presenting the options should improve accuracy on this type."

Respond ONLY with valid JSON matching the schema below. No markdown, no preamble.

Schema:
{
  "accuracy_by_type": {"type_name": float, ...},
  "weakest_type": "string",
  "top_failures": [
    {
      "question": "string",
      "gold_answer": "A|B|C|D",
      "predicted_answer": "A|B|C|D|null",
      "raw_output_excerpt": "string",
      "analysis": "string"
    }
  ],
  "hypothesis": "string",
  "summary": "string"
}
"""


def _format_strategy_block(strategy: Strategy) -> str:
    lines = [
        "=== Strategy ===",
        f"ID: {strategy.id[:8]}",
        f"CoT format: {strategy.cot_format.value}",
        f"Few-shot examples: {len(strategy.few_shot_examples)}",
        f"Prompt template:\n{strategy.prompt_template}",
    ]
    if strategy.few_shot_examples:
        lines.append("\nFew-shot examples:")
        for ex in strategy.few_shot_examples:
            lines.append(
                f"  Q: {ex.question[:100]} | A: {ex.answer}"
                + (f" | Reasoning: {ex.reasoning[:100]}" if ex.reasoning else "")
            )
    return "\n".join(lines)


def _format_eval_block(eval_result: EvalResult) -> str:
    lines = [
        "=== Evaluation Results ===",
        f"Split: {eval_result.split}",
        f"Overall accuracy: {eval_result.accuracy:.3f} ({eval_result.num_correct}/{eval_result.num_examples})",
        "\nAccuracy by question type:",
    ]
    for q_type, acc in sorted(eval_result.accuracy_by_type.items(), key=lambda x: x[1]):
        count = eval_result.count_by_type.get(q_type, 0)
        lines.append(f"  {q_type}: {acc:.3f} ({count} examples)")
    return "\n".join(lines)


def _format_failures_block(failures: list[QuestionResult]) -> str:
    lines = ["\n=== Top Failure Cases ==="]
    for i, f in enumerate(failures[:10], start=1):
        lines.append(
            f"\nFailure {i}:\n"
            f"  Type: {f.question_type}\n"
            f"  Question: {f.question[:150]}\n"
            f"  Gold: {f.gold_answer} | Predicted: {f.predicted_answer}\n"
            f"  Model output: {f.raw_output[:200]}"
        )
    return "\n".join(lines)


def _build_user_message(strategy: Strategy, eval_result: EvalResult) -> str:
    failures = eval_result.failures(top_k=10)
    return (
        _format_strategy_block(strategy) + "\n\n"
        + _format_eval_block(eval_result) + "\n\n"
        + _format_failures_block(failures) + "\n\n"
        "=== Your Task ===\n"
        "Analyse these results and produce a structured reflection. "
        "Focus on the weakest question type. "
        "Your hypothesis must directly suggest what the next prompt template should do differently.\n\n"
        "Respond with ONLY valid JSON."
    )


# ------------------------------------------------------------------
# Main reflect function
# ------------------------------------------------------------------


def reflect(
    strategy: Strategy,
    eval_result: EvalResult,
    client: Optional[anthropic.Anthropic] = None,
    model: str = "claude-sonnet-4-20250514",
    max_retries: int = 4,
) -> tuple[Reflection, int]:
    """
    Call Claude to reflect on a strategy's evaluation results.

    Parameters
    ----------
    strategy:
        The strategy that was evaluated.
    eval_result:
        The results from executor.evaluate().
    client:
        An instantiated anthropic.Anthropic client. If None, a new one is
        created using the ANTHROPIC_API_KEY environment variable.
    model:
        Claude model to use.
    max_retries:
        Number of retries on API or validation failure.

    Returns
    -------
    (reflection, total_tokens) — the structured Reflection and tokens used.

    Raises
    ------
    RuntimeError if all retries are exhausted.
    """
    if client is None:
        client = anthropic.Anthropic()

    user_message = _build_user_message(strategy, eval_result)
    last_error: Optional[Exception] = None

    for attempt in range(max_retries):
        if attempt > 0:
            wait = 2 ** attempt
            logger.warning(
                "Reflect attempt %d/%d failed (%s). Retrying in %ds…",
                attempt,
                max_retries,
                last_error,
                wait,
            )
            time.sleep(wait)

        try:
            logger.info(
                "Calling %s for reflection on strategy %s (attempt %d).",
                model,
                strategy.id[:8],
                attempt + 1,
            )
            response = client.messages.create(
                model=model,
                max_tokens=2048,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )

            total_tokens = response.usage.input_tokens + response.usage.output_tokens
            raw_text = response.content[0].text.strip()

            logger.debug("Raw reflection (%d chars): %s…", len(raw_text), raw_text[:300])

            if raw_text.startswith("```"):
                raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
                raw_text = re.sub(r"\s*```$", "", raw_text)

            structured = StructuredReflection.model_validate_json(raw_text)

            # Build failure dicts from the top failures for storage.
            failure_dicts = [
                {
                    "question": f.question,
                    "question_type": _find_question_type(f.question, eval_result),
                    "gold_answer": f.gold_answer,
                    "predicted_answer": f.predicted_answer,
                    "raw_output": f.raw_output_excerpt,
                    "analysis": f.analysis,
                }
                for f in structured.top_failures
            ]

            reflection = Reflection(
                strategy_id=strategy.id,
                accuracy_by_type=structured.accuracy_by_type,
                top_failures=failure_dicts,
                hypothesis=structured.hypothesis,
                summary=structured.summary,
                raw_response=raw_text,
            )

            logger.info(
                "Reflection complete. Weakest type: %s. Tokens: %d.",
                structured.weakest_type,
                total_tokens,
            )
            return reflection, total_tokens

        except (anthropic.APIError, anthropic.APIConnectionError, anthropic.RateLimitError) as exc:
            last_error = exc
            logger.warning("API error on attempt %d: %s", attempt + 1, exc)

        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            logger.warning("JSON/validation error on attempt %d: %s", attempt + 1, exc)

    raise RuntimeError(
        f"Reflector failed after {max_retries} attempts. Last error: {last_error}"
    )


def _find_question_type(question_text: str, eval_result: EvalResult) -> str:
    """Look up the question type from per-question records by matching text."""
    for r in eval_result.per_question:
        if r.question == question_text or r.question[:100] == question_text[:100]:
            return r.question_type
    return "unknown"
