"""
self_reflector.py — Self-optimization reflector: uses Qwen itself as the meta-agent.

Qwen analyses its own failure cases and generates a hypothesis for improvement.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from executor import EvalResult
from strategy import Reflection, Strategy

logger = logging.getLogger(__name__)

_SYSTEM_REFLECT = """\
Bạn là trợ lý nghiên cứu NLP đang phân tích hiệu suất của một chiến lược prompting \
trên bài toán đọc hiểu tiếng Việt (ViMMRC 2.0).

Dựa trên kết quả đánh giá và các lỗi sai, hãy:
1. Xác định loại câu hỏi yếu nhất.
2. Phân tích nguyên nhân thất bại cụ thể.
3. Đề xuất giả thuyết cải thiện có thể kiểm chứng được.

Trả lời CHÍNH XÁC theo JSON sau, không thêm văn bản khác:
{
  "accuracy_by_type": {"tên_loại": float},
  "weakest_type": "tên loại yếu nhất",
  "hypothesis": "giả thuyết cụ thể và có thể kiểm chứng",
  "summary": "tóm tắt ngắn gọn"
}"""


def _build_reflect_message(strategy: Strategy, eval_result: EvalResult) -> str:
    lines = [
        f"=== Chiến lược ===",
        f"CoT: {strategy.cot_format.value}",
        f"Template: {strategy.prompt_template[:300]!r}",
        f"\n=== Kết quả đánh giá ===",
        f"Độ chính xác tổng: {eval_result.accuracy:.3f} ({eval_result.num_correct}/{eval_result.num_examples})",
        "\nĐộ chính xác theo loại câu hỏi:",
    ]
    for q_type, acc in sorted(eval_result.accuracy_by_type.items(), key=lambda x: x[1]):
        count = eval_result.count_by_type.get(q_type, 0)
        lines.append(f"  {q_type}: {acc:.3f} ({count} ví dụ)")

    failures = eval_result.failures(top_k=5)
    lines.append("\n=== Các lỗi sai tiêu biểu ===")
    for i, f in enumerate(failures, 1):
        lines.append(
            f"\nLỗi {i}:\n"
            f"  Câu hỏi: {f.question[:120]}\n"
            f"  Đúng: {f.gold_answer} | Dự đoán: {f.predicted_answer}\n"
            f"  Output: {f.raw_output[:150]}"
        )

    lines.append("\nTrả lời bằng JSON.")
    return "\n".join(lines)


def reflect_self(
    strategy: Strategy,
    eval_result: EvalResult,
    model,  # QwenInference
    max_retries: int = 5,
) -> tuple[Reflection, int]:
    """
    Use the Qwen inference model itself to reflect on evaluation results.

    Parameters
    ----------
    strategy:
        The strategy that was evaluated.
    eval_result:
        Results from executor.evaluate().
    model:
        A loaded QwenInference instance.
    max_retries:
        Retry on parse failure.

    Returns
    -------
    (reflection, estimated_tokens)
    """
    user_message = _build_reflect_message(strategy, eval_result)
    prompt = model.format_prompt(
        system_message=_SYSTEM_REFLECT,
        user_message=user_message,
    )

    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            logger.info("Self-reflect attempt %d/%d for strategy %s.", attempt + 1, max_retries, strategy.id[:8])
            raw = model.generate_text(prompt, max_new_tokens=512, temperature=0.5)
            logger.debug("Raw self-reflect output: %s", raw[:300])

            raw_clean = re.sub(r"^```(?:json)?\s*", "", raw.strip())
            raw_clean = re.sub(r"\s*```$", "", raw_clean)

            match = re.search(r"\{.*\}", raw_clean, re.DOTALL)
            if not match:
                raise ValueError("No JSON object found in output.")
            data = json.loads(match.group())

            accuracy_by_type = {
                str(k): float(v)
                for k, v in data.get("accuracy_by_type", eval_result.accuracy_by_type).items()
            }
            hypothesis = data.get("hypothesis", "").strip()
            summary = data.get("summary", "").strip()

            if not hypothesis:
                raise ValueError("Empty hypothesis in response.")

            reflection = Reflection(
                strategy_id=strategy.id,
                accuracy_by_type=accuracy_by_type or eval_result.accuracy_by_type,
                top_failures=[
                    {
                        "question": r.question,
                        "gold_answer": r.gold_answer,
                        "predicted_answer": r.predicted_answer,
                        "raw_output": r.raw_output[:200],
                    }
                    for r in eval_result.failures(top_k=5)
                ],
                hypothesis=hypothesis,
                summary=summary or f"Dev accuracy: {eval_result.accuracy:.3f}",
                raw_response=raw,
            )
            estimated_tokens = model.count_tokens(prompt) + model.count_tokens(raw)
            logger.info("Self-reflect complete. Hypothesis: %s", hypothesis[:100])
            return reflection, estimated_tokens

        except Exception as exc:
            last_error = exc
            logger.warning("Self-reflect attempt %d failed: %s", attempt + 1, exc)

    # Fallback: build a minimal reflection from eval_result directly.
    logger.error("Self-reflector failed after %d attempts. Using fallback reflection.", max_retries)
    weakest = min(eval_result.accuracy_by_type, key=eval_result.accuracy_by_type.get) if eval_result.accuracy_by_type else "unknown"
    reflection = Reflection(
        strategy_id=strategy.id,
        accuracy_by_type=eval_result.accuracy_by_type,
        top_failures=[
            {"question": r.question, "gold_answer": r.gold_answer, "predicted_answer": r.predicted_answer}
            for r in eval_result.failures(top_k=5)
        ],
        hypothesis=f"Chiến lược hiện tại yếu nhất ở loại '{weakest}'. Cần cải thiện.",
        summary=f"Dev accuracy: {eval_result.accuracy:.3f}. Fallback reflection.",
        raw_response="",
    )
    return reflection, 0
