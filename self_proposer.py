"""
self_proposer.py — Self-optimization proposer: uses Qwen itself as the meta-agent.

This is the academically interesting setup Hoang described: the same model
that does inference also proposes its own improved prompting strategies.
No external API (Gemini/GPT) required — Qwen guides Qwen.

The prompt is simpler than the Gemini version because Qwen-7B is weaker at
following complex JSON schemas. We use a two-pass approach:
  1. Ask Qwen to reason about failures in free text.
  2. Ask Qwen to format a strategy as JSON based on its own reasoning.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Optional

from strategy import (
    CoTFormat,
    FewShotExample,
    RetrievalConfig,
    Strategy,
    StrategyHistory,
    StrategyMetadata,
)

logger = logging.getLogger(__name__)

_VALID_COT = {f.value for f in CoTFormat}

_SYSTEM_PROPOSE = """\
Bạn là một nhà nghiên cứu NLP đang tối ưu hóa chiến lược prompting cho bài toán \
đọc hiểu tiếng Việt (ViMMRC 2.0). Nhiệm vụ của bạn là đề xuất một chiến lược \
prompting mới dựa trên lịch sử các chiến lược đã thử và kết quả phản ánh gần nhất.

Trả lời CHÍNH XÁC theo định dạng JSON sau, không thêm bất kỳ văn bản nào khác:
{
  "hypothesis": "giả thuyết cụ thể về lý do chiến lược trước thất bại",
  "prompt_template": "template Python với các key: {passage}, {question}, {choices}, {few_shot_block}, {cot_instruction}",
  "cot_format": "none hoặc stepbystep hoặc chain",
  "reasoning": "lý do bạn nghĩ chiến lược này sẽ tốt hơn"
}"""


def _build_propose_message(history: StrategyHistory) -> str:
    lines = ["=== Lịch sử chiến lược ==="]
    for s, r in zip(history.strategies, history.reflections):
        acc = f"{s.metadata.dev_accuracy:.3f}" if s.metadata.dev_accuracy is not None else "chưa đánh giá"
        lines.append(
            f"\nIteration {s.metadata.iteration} | dev_accuracy={acc} | cot={s.cot_format.value}"
        )
        lines.append(f"  Template: {s.prompt_template[:300]!r}")
        if r is not None:
            lines.append(f"  Loại câu hỏi yếu nhất: {min(r.accuracy_by_type, key=r.accuracy_by_type.get) if r.accuracy_by_type else 'unknown'}")
            lines.append(f"  Giả thuyết: {r.hypothesis[:200]}")

    next_iter = len(history.strategies)
    lines.append(f"\n=== Nhiệm vụ ===")
    lines.append(f"Đề xuất chiến lược cho iteration {next_iter}. Trả lời bằng JSON.")
    return "\n".join(lines)


def propose_self(
    history: StrategyHistory,
    model,  # QwenInference
    max_retries: int = 5,
) -> tuple[Strategy, int]:
    """
    Use the Qwen inference model itself to propose a new strategy.

    Parameters
    ----------
    history:
        Full strategy history including reflections.
    model:
        A loaded QwenInference instance.
    max_retries:
        Retry on parse failure.

    Returns
    -------
    (strategy, estimated_tokens)
    """
    user_message = _build_propose_message(history)
    prompt = model.format_prompt(
        system_message=_SYSTEM_PROPOSE,
        user_message=user_message,
    )
    next_iteration = len(history.strategies)
    parent_id = history.latest_strategy().id if history.latest_strategy() else None

    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            logger.info("Self-propose attempt %d/%d (iteration %d).", attempt + 1, max_retries, next_iteration)
            raw = model.generate_text(prompt, max_new_tokens=512, temperature=0.7)
            logger.debug("Raw self-propose output: %s", raw[:300])

            # Strip markdown fences.
            raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
            raw = re.sub(r"\s*```$", "", raw)

            # Extract first JSON object.
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                raise ValueError("No JSON object found in output.")
            data = json.loads(match.group())

            hypothesis = data.get("hypothesis", "").strip()
            prompt_template = data.get("prompt_template", "").strip()
            cot_format_str = data.get("cot_format", "none").strip().lower()

            if not prompt_template:
                raise ValueError("Empty prompt_template in response.")
            if cot_format_str not in _VALID_COT:
                cot_format_str = "none"

            # Ensure required template keys are present; fall back to seed template.
            required_keys = ["{passage}", "{question}", "{choices}"]
            if not all(k in prompt_template for k in required_keys):
                raise ValueError(f"prompt_template missing required keys: {required_keys}")

            strategy = Strategy(
                id=str(uuid.uuid4()),
                prompt_template=prompt_template,
                cot_format=CoTFormat(cot_format_str),
                few_shot_examples=[],
                retrieval_config=RetrievalConfig(enabled=False),
                metadata=StrategyMetadata(
                    iteration=next_iteration,
                    parent_id=parent_id,
                ),
            )
            estimated_tokens = model.count_tokens(prompt) + model.count_tokens(raw)
            logger.info(
                "Self-propose accepted: id=%s, cot=%s.", strategy.id[:8], strategy.cot_format.value
            )
            return strategy, estimated_tokens

        except Exception as exc:
            last_error = exc
            logger.warning("Self-propose attempt %d failed: %s", attempt + 1, exc)

    logger.error("Self-proposer failed after %d attempts. Using seed template.", max_retries)
    from strategy import make_seed_strategy
    fallback = make_seed_strategy()
    fallback.metadata.iteration = next_iteration
    fallback.metadata.parent_id = parent_id
    return fallback, 0
