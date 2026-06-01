"""
proposer.py — LLM-driven strategy proposer for EvoAgent.

The proposer calls Claude (claude-sonnet-4-20250514) with a carefully
engineered prompt that includes the full strategy history and the most
recent reflection. It asks Claude to:
  1. Identify the weakest question categories from the last eval.
  2. Form a specific hypothesis about why the current strategy fails there.
  3. Generate a new strategy (prompt template + CoT format + few-shot examples)
     that directly addresses that hypothesis.

Claude's response must be valid JSON matching the ProposedStrategy schema.
We use Pydantic for validation and retry with exponential backoff on failure.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Optional

import anthropic
from pydantic import BaseModel, Field, field_validator

from strategy import (
    CoTFormat,
    FewShotExample,
    RetrievalConfig,
    Strategy,
    StrategyHistory,
    StrategyMetadata,
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Pydantic schema for Claude's structured output
# ------------------------------------------------------------------


class ProposedFewShotExample(BaseModel):
    """One few-shot example as proposed by Claude."""

    passage: str = Field(description="A short Vietnamese reading passage.")
    question: str = Field(description="A multiple-choice question about the passage.")
    choices: dict[str, str] = Field(description='{"A": "...", "B": "...", "C": "...", "D": "..."}')
    answer: str = Field(description='Correct answer letter: "A", "B", "C", or "D".')
    reasoning: Optional[str] = Field(
        default=None,
        description="Optional step-by-step reasoning for this example.",
    )

    @field_validator("answer")
    @classmethod
    def answer_must_be_valid(cls, v: str) -> str:
        v = v.strip().upper()
        if v not in {"A", "B", "C", "D"}:
            raise ValueError(f"answer must be A/B/C/D, got '{v}'")
        return v


class ProposedStrategy(BaseModel):
    """
    The full strategy proposal returned by Claude.

    Claude is instructed to fill every field with concrete values, not
    placeholders. The prompt_template must be a valid Python format string
    using only the keys: passage, question, choices, few_shot_block,
    cot_instruction.
    """

    hypothesis: str = Field(
        description=(
            "A specific, testable hypothesis about why the previous strategy "
            "failed and what this new strategy will fix."
        )
    )
    prompt_template: str = Field(
        description=(
            "A Python format string for the user message. "
            "Must use only these keys: {passage}, {question}, {choices}, "
            "{few_shot_block}, {cot_instruction}."
        )
    )
    cot_format: str = Field(
        description='Chain-of-thought format: "none", "stepbystep", or "chain".'
    )
    few_shot_examples: list[ProposedFewShotExample] = Field(
        default_factory=list,
        description="Zero to three few-shot examples. Keep passages short (≤100 words).",
    )
    reasoning: str = Field(
        description=(
            "Your internal reasoning about why this strategy should outperform "
            "the previous one. This is logged but not shown to the model."
        )
    )

    @field_validator("cot_format")
    @classmethod
    def cot_format_must_be_valid(cls, v: str) -> str:
        valid = {f.value for f in CoTFormat}
        if v not in valid:
            raise ValueError(f"cot_format must be one of {valid}, got '{v}'")
        return v


# ------------------------------------------------------------------
# System and user prompt builders
# ------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert NLP researcher specialising in Vietnamese reading comprehension.
Your task is to design better prompting strategies for a Qwen2.5-7B-Instruct model
that answers multiple-choice questions from the ViMMRC 2.0 dataset.

ViMMRC 2.0 contains passages from Vietnamese literature textbooks (grades 6–12).
Each example has a passage (100–400 words), a question, and four answer choices (A/B/C/D).

You will be given:
  - The history of all strategies tried so far with their dev accuracies.
  - The most recent reflection, including per-category accuracy breakdown and
    a hypothesis about why the current strategy fails.

Your job is to propose ONE new strategy that directly tests a specific hypothesis
for improving accuracy. Think like a scientist: form a clear hypothesis and design
a strategy that isolates and tests it.

IMPORTANT CONSTRAINTS:
  - The prompt_template must be a Python format string using ONLY these keys:
    {passage}, {question}, {choices}, {few_shot_block}, {cot_instruction}
  - Few-shot passages must be in Vietnamese and ≤100 words each.
  - Do not propose a strategy identical to one already tried.
  - Be specific: vague strategies like "add more detail" are not helpful.

Respond ONLY with a valid JSON object matching the schema below. No markdown,
no preamble, no explanation outside the JSON.

Schema:
{
  "hypothesis": "string — specific testable hypothesis",
  "prompt_template": "string — Python format string",
  "cot_format": "none | stepbystep | chain",
  "few_shot_examples": [
    {
      "passage": "string",
      "question": "string",
      "choices": {"A": "string", "B": "string", "C": "string", "D": "string"},
      "answer": "A|B|C|D",
      "reasoning": "string or null"
    }
  ],
  "reasoning": "string — your internal reasoning"
}
"""


def _build_history_block(history: StrategyHistory) -> str:
    """Format the strategy history into a concise, information-dense block."""
    if not history.strategies:
        return "No strategies have been evaluated yet. This is the first proposal."

    lines = ["=== Strategy History ==="]
    for i, (s, r) in enumerate(zip(history.strategies, history.reflections)):
        acc = (
            f"{s.metadata.dev_accuracy:.3f}"
            if s.metadata.dev_accuracy is not None
            else "not evaluated"
        )
        lines.append(
            f"\nIteration {s.metadata.iteration} | id={s.id[:8]} | "
            f"dev_accuracy={acc} | cot={s.cot_format.value} | "
            f"few_shot={len(s.few_shot_examples)}"
        )
        lines.append(f"  Template (first 200 chars): {s.prompt_template[:200]!r}")
        if r is not None:
            lines.append(f"  Hypothesis that guided this strategy: {r.hypothesis[:300]}")
            lines.append(f"  Accuracy by type: {r.accuracy_by_type}")
    return "\n".join(lines)


def _build_reflection_block(history: StrategyHistory) -> str:
    """Format the most recent reflection."""
    ref = history.latest_reflection()
    if ref is None:
        return "No reflections available yet."

    lines = [
        "=== Most Recent Reflection ===",
        f"Strategy: {ref.strategy_id[:8]}",
        f"Accuracy by type: {json.dumps(ref.accuracy_by_type, ensure_ascii=False)}",
        f"\nTop failure cases:",
    ]
    for fc in ref.top_failures[:5]:
        lines.append(
            f"  Q: {fc.get('question', '')[:120]}\n"
            f"  Gold: {fc.get('gold_answer')} | Predicted: {fc.get('predicted_answer')}\n"
            f"  Output: {str(fc.get('raw_output', ''))[:200]}"
        )
    lines.append(f"\nHypothesis: {ref.hypothesis}")
    lines.append(f"\nSummary: {ref.summary}")
    return "\n".join(lines)


def _build_user_message(history: StrategyHistory) -> str:
    history_block = _build_history_block(history)
    reflection_block = _build_reflection_block(history)
    next_iteration = len(history.strategies)

    return (
        f"{history_block}\n\n"
        f"{reflection_block}\n\n"
        f"=== Your Task ===\n"
        f"Propose strategy for iteration {next_iteration}. "
        f"Target the weakest question category identified in the reflection. "
        f"Your hypothesis must be specific and falsifiable.\n\n"
        f"Respond with ONLY valid JSON."
    )


# ------------------------------------------------------------------
# Main propose function
# ------------------------------------------------------------------


def propose(
    history: StrategyHistory,
    client: Optional[anthropic.Anthropic] = None,
    model: str = "claude-sonnet-4-20250514",
    max_retries: int = 4,
) -> tuple[Strategy, int]:
    """
    Call Claude to propose a new strategy based on the history and latest reflection.

    Parameters
    ----------
    history:
        The full strategy history including reflections.
    client:
        An instantiated anthropic.Anthropic client. If None, one is created
        using the ANTHROPIC_API_KEY environment variable.
    model:
        Claude model to use for proposing.
    max_retries:
        Number of retries on API or validation failure (exponential backoff).

    Returns
    -------
    (strategy, total_tokens) where total_tokens is input + output tokens used.

    Raises
    ------
    RuntimeError if all retries are exhausted.
    """
    if client is None:
        client = anthropic.Anthropic()

    user_message = _build_user_message(history)
    next_iteration = len(history.strategies)
    parent_id = history.latest_strategy().id if history.latest_strategy() else None

    last_error: Optional[Exception] = None

    for attempt in range(max_retries):
        if attempt > 0:
            wait = 2 ** attempt  # 2, 4, 8 seconds
            logger.warning(
                "Propose attempt %d/%d failed (%s). Retrying in %ds…",
                attempt,
                max_retries,
                last_error,
                wait,
            )
            time.sleep(wait)

        try:
            logger.info(
                "Calling %s for strategy proposal (iteration %d, attempt %d).",
                model,
                next_iteration,
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

            logger.debug("Raw Claude response (%d chars): %s…", len(raw_text), raw_text[:300])

            # Strip any accidental markdown fencing.
            if raw_text.startswith("```"):
                raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
                raw_text = re.sub(r"\s*```$", "", raw_text)

            proposed = ProposedStrategy.model_validate_json(raw_text)
            strategy = _proposed_to_strategy(proposed, next_iteration, parent_id)

            logger.info(
                "Proposal accepted: id=%s, cot=%s, few_shot=%d, tokens=%d.",
                strategy.id[:8],
                strategy.cot_format.value,
                len(strategy.few_shot_examples),
                total_tokens,
            )
            return strategy, total_tokens

        except (anthropic.APIError, anthropic.APIConnectionError, anthropic.RateLimitError) as exc:
            last_error = exc
            logger.warning("API error on attempt %d: %s", attempt + 1, exc)

        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            logger.warning("JSON/validation error on attempt %d: %s", attempt + 1, exc)

    raise RuntimeError(
        f"Proposer failed after {max_retries} attempts. Last error: {last_error}"
    )


def _proposed_to_strategy(
    proposed: ProposedStrategy,
    iteration: int,
    parent_id: Optional[str],
) -> Strategy:
    """Convert a validated ProposedStrategy Pydantic model into a Strategy dataclass."""
    few_shot = [
        FewShotExample(
            passage=ex.passage,
            question=ex.question,
            choices=ex.choices,
            answer=ex.answer,
            reasoning=ex.reasoning,
        )
        for ex in proposed.few_shot_examples
    ]
    return Strategy(
        id=str(uuid.uuid4()),
        prompt_template=proposed.prompt_template,
        cot_format=CoTFormat(proposed.cot_format),
        few_shot_examples=few_shot,
        retrieval_config=RetrievalConfig(enabled=False),
        metadata=StrategyMetadata(
            iteration=iteration,
            parent_id=parent_id,
        ),
    )


