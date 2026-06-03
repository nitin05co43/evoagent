"""
model.py — QwenInference: vLLM-based inference wrapper for Qwen2.5-7B-Instruct.

Uses vLLM for high-throughput batched inference. vLLM is 5-10x faster than
the HuggingFace generate() pipeline on the same hardware.

For T4 (16 GB VRAM), use an AWQ-quantized model:
    Qwen/Qwen2.5-7B-Instruct-AWQ  (~4 GB weights, leaves room for KV cache)

Answer extraction handles all common Qwen output formats:
    "Đáp án: B", "(B)", "Answer: B", bare "B", etc.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Regex patterns for extracting a single A/B/C/D answer from model output.
# Listed from most specific to least specific; first match wins.
_ANSWER_PATTERNS = [
    # "Answer: A" or "Đáp án: B" (English or Vietnamese label)
    re.compile(r"(?:answer|đáp\s*án)\s*[:\-]\s*\(?([ABCD])\)?", re.IGNORECASE),
    # "the answer is A"
    re.compile(r"the\s+answer\s+is\s+\(?([ABCD])\)?", re.IGNORECASE),
    # "(A)" — parenthesised letter
    re.compile(r"\(([ABCD])\)"),
    # A bare capital letter on its own line or at end of string
    re.compile(r"(?:^|[\s\n])([ABCD])(?:\s*$|\s*[\.\,\n])"),
    # Last-resort: any lone A/B/C/D in the output
    re.compile(r"\b([ABCD])\b"),
]


@dataclass
class GenerationResult:
    """Holds the raw text output and token counts for a single example."""

    raw_output: str
    predicted_answer: Optional[str]  # "A", "B", "C", "D", or None
    input_tokens: int
    output_tokens: int


class QwenInference:
    """
    vLLM-backed inference wrapper for Qwen2.5-Instruct models.

    Parameters
    ----------
    model_name_or_path:
        HuggingFace model ID or local path. For T4 GPUs use the AWQ variant:
        "Qwen/Qwen2.5-7B-Instruct-AWQ"
    batch_size:
        Ignored — vLLM manages batching internally. Kept for API compatibility.
    max_new_tokens:
        Maximum tokens generated per example.
    temperature:
        Sampling temperature. 0.0 = greedy decoding.
    use_4bit:
        If True and the model name doesn't already indicate quantization,
        logs a warning. For T4, pass an AWQ/GPTQ model name instead.
    gpu_memory_utilization:
        Fraction of GPU memory vLLM may use for weights + KV cache (0–1).
    max_model_len:
        Maximum sequence length (prompt + completion). Longer sequences are
        truncated. 2048 is sufficient for ViMMRC passages.
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct-AWQ",
        batch_size: int = 8,          # unused by vLLM; kept for compatibility
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        use_4bit: bool = True,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 4096,
    ):
        self.model_name_or_path = model_name_or_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.use_4bit = use_4bit
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len

        self._llm = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the model via vLLM. Call once before inference."""
        from vllm import LLM, SamplingParams  # noqa: F401 (validate import early)
        from transformers import AutoTokenizer

        name = self.model_name_or_path
        name_lower = name.lower()

        # Detect quantization from model name.
        if "awq" in name_lower:
            quantization = "awq"
        elif "gptq" in name_lower:
            quantization = "gptq"
        elif "int4" in name_lower or "int8" in name_lower:
            quantization = "gptq"
        else:
            quantization = None
            if self.use_4bit:
                logger.warning(
                    "use_4bit=True but model '%s' does not appear to be a "
                    "quantized checkpoint. Consider using "
                    "'Qwen/Qwen2.5-7B-Instruct-AWQ' for T4 GPUs.",
                    name,
                )

        logger.info(
            "Loading vLLM engine: model=%s quantization=%s gpu_mem=%.0f%%",
            name,
            quantization or "none (fp16)",
            self.gpu_memory_utilization * 100,
        )

        self._llm = LLM(
            model=name,
            quantization=quantization,
            dtype="half",
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            trust_remote_code=True,
        )

        # Load tokenizer separately for apply_chat_template / count_tokens.
        self._tokenizer = AutoTokenizer.from_pretrained(
            name,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        logger.info("Model loaded successfully.")

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def generate_batch(self, prompts: list[str]) -> list[GenerationResult]:
        """
        Run generation for a list of already-formatted prompt strings.

        vLLM handles batching and scheduling internally — pass all prompts
        at once for maximum throughput.
        """
        if self._llm is None:
            raise RuntimeError("Call load() before generate_batch().")

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=self.temperature if self.temperature > 0.0 else 0.0,
            # vLLM requires top_p < 1.0 when temperature > 0
            top_p=0.95 if self.temperature > 0.0 else 1.0,
        )

        outputs = self._llm.generate(prompts, sampling_params)

        results: list[GenerationResult] = []
        for req_output in outputs:
            completion = req_output.outputs[0]
            raw_text = completion.text.strip()
            predicted = extract_answer(raw_text)
            input_tokens = len(req_output.prompt_token_ids)
            output_tokens = len(completion.token_ids)
            results.append(
                GenerationResult(
                    raw_output=raw_text,
                    predicted_answer=predicted,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            )

        return results

    def format_prompt(self, system_message: str, user_message: str) -> str:
        """
        Build a ChatML-formatted string for Qwen2.5-Instruct using the
        tokenizer's apply_chat_template().
        """
        if self._tokenizer is None:
            raise RuntimeError("Call load() before format_prompt().")
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def count_tokens(self, text: str) -> int:
        if self._tokenizer is None:
            raise RuntimeError("Call load() before count_tokens().")
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None


# ------------------------------------------------------------------
# Answer extraction (module-level for testability)
# ------------------------------------------------------------------


def extract_answer(text: str) -> Optional[str]:
    """
    Extract a single A/B/C/D answer letter from a model's free-form output.

    Tries patterns from most specific to least specific. Returns the uppercase
    letter, or None if no answer can be reliably extracted.

    Examples:
        "A"               -> "A"
        "(B)"             -> "B"
        "Answer: C"       -> "C"
        "the answer is D" -> "D"
        "Đáp án: B"       -> "B"
        "...therefore B." -> "B"
    """
    if not text:
        return None
    for pattern in _ANSWER_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).upper()
    return None


def build_choices_block(choices: dict[str, str]) -> str:
    """
    Format a choices dict into the multi-line block shown in prompts.

    Input:  {"A": "Hà Nội", "B": "Đà Nẵng", "C": "Huế", "D": "Hội An"}
    Output: "A. Hà Nội\\nB. Đà Nẵng\\nC. Huế\\nD. Hội An"
    """
    return "\n".join(
        f"{letter}. {choices[letter]}"
        for letter in ["A", "B", "C", "D"]
        if letter in choices
    )
