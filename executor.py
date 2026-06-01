"""
executor.py — Evaluate a Strategy on a dataset split.

The executor translates a Strategy into per-example prompts, runs batched
inference with QwenInference, collects predicted vs. gold answers, and returns
an EvalResult with accuracy broken down by question type (if available).

Question-type bucketing: ViMMRC 2.0 does not always include an explicit type
field. We attempt lightweight heuristic classification on the question text
when a structured field is absent.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from datasets import Dataset

from model import QwenInference, build_choices_block, extract_answer
from strategy import CoTFormat, FewShotExample, Strategy

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Result types
# ------------------------------------------------------------------


@dataclass
class QuestionResult:
    """Per-question prediction record stored for the reflector."""

    question_id: str           # Row index as string (or dataset-provided id)
    passage: str
    question: str
    choices: dict[str, str]
    gold_answer: str           # "A", "B", "C", or "D"
    predicted_answer: Optional[str]  # None if extraction failed
    is_correct: bool
    raw_output: str
    question_type: str = "unknown"  # Heuristic or dataset-provided category
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class EvalResult:
    """Aggregate evaluation results for one strategy on one dataset split."""

    strategy_id: str
    split: str
    num_examples: int
    num_correct: int
    accuracy: float
    accuracy_by_type: dict[str, float] = field(default_factory=dict)
    # Counts per type for weighted averaging
    count_by_type: dict[str, int] = field(default_factory=dict)
    per_question: list[QuestionResult] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "strategy_id": self.strategy_id,
            "split": self.split,
            "num_examples": self.num_examples,
            "num_correct": self.num_correct,
            "accuracy": self.accuracy,
            "accuracy_by_type": self.accuracy_by_type,
            "count_by_type": self.count_by_type,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "elapsed_seconds": self.elapsed_seconds,
            # per_question is intentionally excluded here to keep the dict compact;
            # the harness saves it separately if needed.
        }

    def failures(self, top_k: int = 10) -> list[QuestionResult]:
        """Return up to top_k incorrect predictions."""
        return [r for r in self.per_question if not r.is_correct][:top_k]


# ------------------------------------------------------------------
# Question-type heuristics
# ------------------------------------------------------------------

# Simple keyword mappings for Vietnamese reading comprehension question types.
_TYPE_KEYWORDS: list[tuple[str, list[str]]] = [
    ("main_idea", ["chủ đề", "nội dung chính", "ý chính", "chủ đề chính"]),
    ("inference", ["suy ra", "ngụ ý", "hàm ý", "tác giả muốn nói"]),
    ("detail", ["chi tiết", "theo đoạn", "đoạn văn cho biết", "câu nào"]),
    ("vocabulary", ["từ", "nghĩa của", "hiểu là", "đồng nghĩa", "từ ngữ"]),
    ("cause_effect", ["nguyên nhân", "hậu quả", "vì sao", "tại sao", "kết quả"]),
    ("title", ["nhan đề", "tiêu đề", "tên bài", "đặt tên"]),
]


def classify_question_type(question: str) -> str:
    """
    Heuristically assign a question type from the Vietnamese question text.

    Returns one of: main_idea, inference, detail, vocabulary, cause_effect,
    title, or "other".
    """
    q_lower = question.lower()
    for q_type, keywords in _TYPE_KEYWORDS:
        if any(kw in q_lower for kw in keywords):
            return q_type
    return "other"


# ------------------------------------------------------------------
# Prompt construction
# ------------------------------------------------------------------

_COT_INSTRUCTIONS: dict[CoTFormat, str] = {
    CoTFormat.NONE: "",
    CoTFormat.STEPBYSTEP: (
        "Hãy suy nghĩ từng bước trước khi đưa ra đáp án. "
        "Cuối cùng, viết 'Đáp án: X' với X là chữ cái đúng."
    ),
    CoTFormat.CHAIN: (
        "Hãy lập luận theo chuỗi suy nghĩ (chain-of-thought) để giải thích "
        "lý do chọn đáp án. Kết thúc bằng 'Đáp án: X' với X là A, B, C hoặc D."
    ),
}

_SYSTEM_MESSAGE = (
    "Bạn là một trợ lý AI giỏi đọc hiểu tiếng Việt. "
    "Nhiệm vụ của bạn là trả lời câu hỏi trắc nghiệm dựa trên đoạn văn được cung cấp."
)


def build_few_shot_block(examples: list[FewShotExample]) -> str:
    """
    Format a list of few-shot examples into a single string block.

    Each example is rendered as a complete passage/question/choices/answer unit
    so the model sees what a correct response looks like.
    """
    if not examples:
        return ""

    parts = ["Dưới đây là một số ví dụ mẫu:\n"]
    for idx, ex in enumerate(examples, start=1):
        choices_block = build_choices_block(ex.choices)
        reasoning_part = f"\nGiải thích: {ex.reasoning}" if ex.reasoning else ""
        parts.append(
            f"Ví dụ {idx}:\n"
            f"Đoạn văn: {ex.passage}\n"
            f"Câu hỏi: {ex.question}\n"
            f"{choices_block}\n"
            f"Đáp án: {ex.answer}{reasoning_part}\n"
        )
    parts.append("---\nBây giờ hãy trả lời câu hỏi dưới đây:\n")
    return "\n".join(parts)


def build_prompt(
    strategy: Strategy,
    passage: str,
    question: str,
    choices: dict[str, str],
    tokenizer_fn=None,
) -> str:
    """
    Construct the user-facing portion of the prompt for one example.

    Uses the strategy's prompt_template with the following substitutions:
        {passage}           — the reading passage
        {question}          — the question text
        {choices}           — formatted A/B/C/D options
        {few_shot_block}    — rendered few-shot examples (may be empty)
        {cot_instruction}   — chain-of-thought instruction (may be empty)
    """
    choices_block = build_choices_block(choices)
    few_shot_block = build_few_shot_block(strategy.few_shot_examples)
    cot_instruction = _COT_INSTRUCTIONS[strategy.cot_format]

    try:
        user_message = strategy.prompt_template.format(
            passage=passage,
            question=question,
            choices=choices_block,
            few_shot_block=few_shot_block,
            cot_instruction=cot_instruction,
        )
    except KeyError as exc:
        raise ValueError(
            f"Strategy {strategy.id} prompt_template references unknown key {exc}. "
            "Valid keys: passage, question, choices, few_shot_block, cot_instruction."
        ) from exc

    return user_message


# ------------------------------------------------------------------
# Main evaluation function
# ------------------------------------------------------------------


def evaluate(
    strategy: Strategy,
    split: str,
    dataset: Dataset,
    model: QwenInference,
) -> EvalResult:
    """
    Evaluate a strategy on a HuggingFace Dataset split.

    Parameters
    ----------
    strategy:
        The strategy to evaluate.
    split:
        Human-readable label for logging (e.g., "dev", "train_subset").
    dataset:
        A HuggingFace Dataset with columns: passage, question, choices (list or
        dict), answer. The answer column should contain "A", "B", "C", or "D".
    model:
        A loaded QwenInference instance.

    Returns
    -------
    EvalResult with per-question records and aggregate accuracy stats.
    """
    logger.info(
        "Evaluating strategy %s on %s split (%d examples).",
        strategy.id[:8],
        split,
        len(dataset),
    )
    start_time = time.time()

    # 1. Build all prompts up front so we can batch them efficiently.
    formatted_prompts: list[str] = []
    metadata: list[dict[str, Any]] = []

    for idx, row in enumerate(dataset):
        passage, question, choices, gold = _parse_row(row, idx)
        q_type = (
            row.get("question_type") or classify_question_type(question)
        )

        user_message = build_prompt(strategy, passage, question, choices)
        full_prompt = model.format_prompt(
            system_message=_SYSTEM_MESSAGE,
            user_message=user_message,
        )
        formatted_prompts.append(full_prompt)
        metadata.append(
            {
                "idx": idx,
                "passage": passage,
                "question": question,
                "choices": choices,
                "gold": gold,
                "q_type": q_type,
            }
        )

    # 2. Run batched generation.
    logger.info("Running inference on %d examples…", len(formatted_prompts))
    gen_results = model.generate_batch(formatted_prompts)

    # 3. Collect per-question results.
    per_question: list[QuestionResult] = []
    for meta, gen in zip(metadata, gen_results):
        is_correct = (
            gen.predicted_answer is not None
            and gen.predicted_answer == meta["gold"]
        )
        per_question.append(
            QuestionResult(
                question_id=str(meta["idx"]),
                passage=meta["passage"],
                question=meta["question"],
                choices=meta["choices"],
                gold_answer=meta["gold"],
                predicted_answer=gen.predicted_answer,
                is_correct=is_correct,
                raw_output=gen.raw_output,
                question_type=meta["q_type"],
                input_tokens=gen.input_tokens,
                output_tokens=gen.output_tokens,
            )
        )

    # 4. Compute aggregate accuracy.
    num_correct = sum(r.is_correct for r in per_question)
    total = len(per_question)
    accuracy = num_correct / total if total > 0 else 0.0

    # 5. Accuracy by question type.
    accuracy_by_type: dict[str, float] = {}
    count_by_type: dict[str, int] = {}
    type_correct: dict[str, int] = {}

    for r in per_question:
        q_type = r.question_type
        count_by_type[q_type] = count_by_type.get(q_type, 0) + 1
        type_correct[q_type] = type_correct.get(q_type, 0) + (1 if r.is_correct else 0)

    for q_type, count in count_by_type.items():
        accuracy_by_type[q_type] = type_correct[q_type] / count if count > 0 else 0.0

    total_input = sum(r.input_tokens for r in per_question)
    total_output = sum(r.output_tokens for r in per_question)
    elapsed = time.time() - start_time

    result = EvalResult(
        strategy_id=strategy.id,
        split=split,
        num_examples=total,
        num_correct=num_correct,
        accuracy=accuracy,
        accuracy_by_type=accuracy_by_type,
        count_by_type=count_by_type,
        per_question=per_question,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        elapsed_seconds=elapsed,
    )

    logger.info(
        "Evaluation complete: accuracy=%.3f (%d/%d) in %.1fs. "
        "Tokens: input=%d, output=%d.",
        accuracy,
        num_correct,
        total,
        elapsed,
        total_input,
        total_output,
    )
    return result


# ------------------------------------------------------------------
# Row parsing helpers
# ------------------------------------------------------------------


def _parse_row(row: dict, idx: int) -> tuple[str, str, dict[str, str], str]:
    """
    Normalise a dataset row into (passage, question, choices_dict, gold_letter).

    ViMMRC 2.0 stores choices as a list ["option1", ...] and the answer as an
    index 0–3 or as a letter "A"–"D". This function normalises both formats.
    """
    passage = row.get("context") or row.get("passage") or row.get("article") or ""
    question = row.get("question") or ""

    # Choices: may be a list or a dict.
    raw_choices = row.get("options") or row.get("choices") or {}
    if isinstance(raw_choices, (list, tuple)):
        letters = ["A", "B", "C", "D"]
        choices = {letters[i]: str(v) for i, v in enumerate(raw_choices) if i < 4}
    elif isinstance(raw_choices, dict):
        choices = {k.upper(): str(v) for k, v in raw_choices.items()}
    else:
        choices = {}

    # Answer: may be "A" or an integer index.
    raw_answer = row.get("answer") or row.get("label") or ""
    if isinstance(raw_answer, int):
        letters = ["A", "B", "C", "D"]
        gold = letters[raw_answer] if 0 <= raw_answer < 4 else "A"
    elif isinstance(raw_answer, str):
        # Strip noise: "A)", "a.", "(A)" -> "A"
        cleaned = re.sub(r"[^A-Da-d]", "", raw_answer)
        gold = cleaned[0].upper() if cleaned else "A"
    else:
        gold = "A"

    return passage, question, choices, gold


