"""
reflector.py — LLM-driven reflection on strategy performance.

Your job: call a meta-agent LLM with the evaluation results and return a
structured Reflection that will guide the next propose() call.

The reflection should include:
  - Accuracy broken down by question type
  - Analysis of the top failure cases
  - A concrete, falsifiable hypothesis for the next iteration

TODO: Implement the reflect() function below.
"""

from __future__ import annotations

import logging
from typing import Optional

from executor import EvalResult
from strategy import Reflection, Strategy

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# TODO 1 — Build the meta-agent prompt
# ------------------------------------------------------------------

def _build_system_prompt() -> str:
    """
    Return the system prompt for the reflector meta-agent.

    The prompt should instruct the LLM to act as an NLP research assistant
    diagnosing why a prompting strategy failed on ViMMRC 2.0.

    Key instruction: hypotheses must be specific and actionable.

    Bad: "The model needs to understand the passage better."
    Good: "The model fails on cause_effect questions because the prompt does
           not ask it to identify causal chains. A prompt that explicitly asks
           'What caused X?' before presenting options should help."

    The LLM must respond with valid JSON only. Schema:
    {
      "accuracy_by_type": {"type_name": float, ...},
      "weakest_type": "string",
      "top_failures": [
        {
          "question": "string",
          "gold_answer": "A|B|C|D",
          "predicted_answer": "A|B|C|D or null",
          "raw_output_excerpt": "string",
          "analysis": "string"
        }
      ],
      "hypothesis": "string",
      "summary": "string"
    }
    """
    # TODO: Write and return your system prompt string.
    raise NotImplementedError


def _build_user_message(strategy: Strategy, eval_result: EvalResult) -> str:
    """
    Build the user message for the reflector.

    Should include:
      - Strategy summary (prompt template, CoT format, few-shot count)
      - Overall accuracy and accuracy by question type
      - Top 10 failure cases with the model's raw output

    Hint: use eval_result.accuracy_by_type, eval_result.count_by_type,
    eval_result.failures(top_k=10)
    """
    # TODO: Build and return the user message string.
    raise NotImplementedError


# ------------------------------------------------------------------
# TODO 2 — Parse the LLM response into a Reflection object
# ------------------------------------------------------------------

def _parse_response(raw_text: str, strategy_id: str) -> Reflection:
    """
    Parse the raw JSON response from the meta-agent into a Reflection.

    Steps:
      1. Strip markdown code fences if present
      2. Parse JSON
      3. Validate required fields
      4. Build and return a Reflection object

    Raises ValueError on invalid JSON or missing fields.
    """
    import json
    import re

    # TODO: Implement parsing logic.
    raise NotImplementedError


# ------------------------------------------------------------------
# TODO 3 — Main reflect() function
# ------------------------------------------------------------------

def reflect(
    strategy: Strategy,
    eval_result: EvalResult,
    api_key: Optional[str] = None,
    model: str = "gemini-2.0-flash",
    max_retries: int = 5,
) -> tuple[Reflection, int]:
    """
    Call the meta-agent LLM to reflect on a strategy's evaluation results.

    Parameters
    ----------
    strategy:
        The strategy that was evaluated.
    eval_result:
        The results from executor.evaluate().
    api_key:
        LLM API key. If None, read from environment (e.g. GOOGLE_API_KEY).
    model:
        Which model to call.
    max_retries:
        Retry on API error or invalid JSON (exponential backoff).

    Returns
    -------
    (reflection, total_tokens)
        reflection: structured Reflection object
        total_tokens: estimated token count for budget tracking

    Raises
    ------
    RuntimeError if all retries are exhausted.

    Implementation steps:
      1. Build system prompt and user message
      2. Call the LLM API
      3. Parse the response into a Reflection via _parse_response()
      4. Return (reflection, token_count)
    """
    # TODO: Implement the reflect() function.
    raise NotImplementedError
