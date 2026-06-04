"""
proposer.py — LLM-driven strategy proposer for EvoAgent.

Your job: call a meta-agent LLM (e.g. Gemini, GPT-4o) with the full strategy
history and the most recent reflection, and return a new Strategy.

The meta-agent should:
  1. Identify the weakest question categories from the last evaluation.
  2. Form a specific, testable hypothesis about why the current strategy fails.
  3. Generate a new strategy (prompt template + CoT format + few-shot examples)
     that directly addresses that hypothesis.

TODO: Implement the propose() function below.
"""

from __future__ import annotations

import logging
from typing import Optional

from strategy import Strategy, StrategyHistory

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# TODO 1 — Build the meta-agent prompt
# ------------------------------------------------------------------

def _build_system_prompt() -> str:
    """
    Return the system prompt for the meta-agent.

    The prompt should instruct the LLM to act as an NLP researcher
    designing prompting strategies for Vietnamese reading comprehension.

    Constraints to include:
      - The prompt_template must be a Python format string using ONLY these keys:
        {passage}, {question}, {choices}, {few_shot_block}, {cot_instruction}
      - Few-shot passages must be in Vietnamese and <= 100 words each.
      - The LLM must respond with valid JSON only (no markdown, no preamble).

    The JSON schema the LLM must follow:
    {
      "hypothesis": "string",
      "prompt_template": "string",
      "cot_format": "none | stepbystep | chain",
      "few_shot_examples": [
        {
          "passage": "string",
          "question": "string",
          "choices": {"A": "...", "B": "...", "C": "...", "D": "..."},
          "answer": "A|B|C|D",
          "reasoning": "string or null"
        }
      ],
      "reasoning": "string"
    }
    """
    # TODO: Write and return your system prompt string.
    raise NotImplementedError


def _build_user_message(history: StrategyHistory) -> str:
    """
    Build the user message for the meta-agent.

    Should include:
      - A summary of all strategies tried so far (iteration, dev accuracy,
        CoT format, prompt template excerpt)
      - The most recent reflection (accuracy by question type, top failure
        cases, hypothesis)
      - A clear instruction to propose the next strategy

    Hint: use history.strategies, history.reflections, history.latest_reflection()
    """
    # TODO: Build and return the user message string.
    raise NotImplementedError


# ------------------------------------------------------------------
# TODO 2 — Parse the LLM response into a Strategy object
# ------------------------------------------------------------------

def _parse_response(raw_text: str, iteration: int, parent_id: Optional[str]) -> Strategy:
    """
    Parse the raw JSON response from the meta-agent into a Strategy.

    Steps:
      1. Strip markdown code fences if present (```json ... ```)
      2. Parse JSON
      3. Validate required fields (hypothesis, prompt_template, cot_format)
      4. Build and return a Strategy using the strategy module helpers

    Raises ValueError on invalid JSON or missing fields.

    Useful imports already available in strategy.py:
      Strategy, StrategyMetadata, CoTFormat, FewShotExample, RetrievalConfig
    """
    import json
    import re
    import uuid
    from strategy import CoTFormat, FewShotExample, RetrievalConfig, StrategyMetadata

    # TODO: Implement parsing logic.
    raise NotImplementedError


# ------------------------------------------------------------------
# TODO 3 — Main propose() function
# ------------------------------------------------------------------

def propose(
    history: StrategyHistory,
    api_key: Optional[str] = None,
    model: str = "gemini-2.0-flash",
    max_retries: int = 5,
) -> tuple[Strategy, int]:
    """
    Call the meta-agent LLM to propose a new prompting strategy.

    Parameters
    ----------
    history:
        Full strategy history including all reflections.
    api_key:
        LLM API key. If None, read from environment (e.g. GOOGLE_API_KEY).
    model:
        Which model to call (e.g. "gemini-2.0-flash", "gpt-4o-mini").
    max_retries:
        Retry on API error or invalid JSON response (exponential backoff).

    Returns
    -------
    (strategy, total_tokens)
        strategy: the new Strategy object ready to be evaluated
        total_tokens: estimated token count for this call (for budget tracking)

    Raises
    ------
    RuntimeError if all retries are exhausted.

    Implementation steps:
      1. Build system prompt and user message
      2. Call the LLM API
      3. Parse the response into a Strategy via _parse_response()
      4. Return (strategy, token_count)

    Hint: wrap the call in a retry loop with exponential backoff.
    Use logger.info() to log each attempt.
    """
    # TODO: Implement the propose() function.
    raise NotImplementedError
