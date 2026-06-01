"""
model.py — QwenInference: batched inference wrapper for Qwen2.5-7B-Instruct.

Handles model loading, ChatML prompt construction, batched generation, and
answer extraction from free-form Vietnamese text. Designed to run within the
16 GB VRAM budget of a Kaggle T4 GPU using 4-bit quantization.

Answer extraction is intentionally robust: the model sometimes writes
"Đáp án: B", "(B)", "Answer: B", or just "B" — we handle all of these.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)

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
    predicted_answer: Optional[str]  # "A", "B", "C", "D", or None if extraction failed
    input_tokens: int
    output_tokens: int


class QwenInference:
    """
    Wrapper around Qwen2.5-7B-Instruct for batched multiple-choice inference.

    Parameters
    ----------
    model_name_or_path:
        HuggingFace model ID or local path. Default is the 7B instruct variant.
    batch_size:
        Number of examples to process in a single forward pass. Reduce if you
        hit OOM errors. 8 works reliably on a T4 16 GB with 4-bit quant.
    max_new_tokens:
        Maximum tokens generated per example. 256 is enough for CoT answers;
        increase to 512 if using longer chain-of-thought formats.
    temperature:
        Sampling temperature. 0.0 means greedy decoding (recommended for
        deterministic evaluation).
    use_4bit:
        Whether to load in 4-bit NF4 quantization. Required to fit within 16 GB.
    device:
        "cuda", "cpu", or "auto". "auto" picks CUDA if available.
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct",
        batch_size: int = 8,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        use_4bit: bool = True,
        device: str = "auto",
    ):
        self.model_name_or_path = model_name_or_path
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.use_4bit = use_4bit

        self._device = self._resolve_device(device)
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the tokenizer and model. Call once before inference."""
        logger.info("Loading tokenizer from %s", self.model_name_or_path)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            padding_side="left",  # Left-padding for batched decoder-only generation
        )
        # Qwen2.5 uses EOS as pad by default; set explicitly so the tokenizer
        # does not warn about it on every batch.
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        quantization_config = None
        if self.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            logger.info("Using 4-bit NF4 quantization (T4-friendly).")

        logger.info("Loading model weights — this may take a minute on first run…")
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            quantization_config=quantization_config,
            device_map="auto" if self._device == "cuda" else self._device,
            trust_remote_code=True,
            torch_dtype=torch.float16 if not self.use_4bit else None,
        )
        self._model.eval()
        logger.info("Model loaded successfully.")

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def generate_batch(self, prompts: list[str]) -> list[GenerationResult]:
        """
        Run batched generation for a list of already-formatted prompt strings.

        Each prompt should be a complete ChatML-formatted string as produced by
        format_prompt(). Returns one GenerationResult per prompt.
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Call load() before generate_batch().")

        results: list[GenerationResult] = []

        for batch_start in range(0, len(prompts), self.batch_size):
            batch = prompts[batch_start : batch_start + self.batch_size]
            logger.debug(
                "Generating for batch %d–%d of %d",
                batch_start,
                batch_start + len(batch) - 1,
                len(prompts),
            )

            encodings = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self._model.device)

            input_lengths = encodings["attention_mask"].sum(dim=1).tolist()

            generation_config = GenerationConfig(
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0.0,
                temperature=self.temperature if self.temperature > 0.0 else None,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

            with torch.no_grad():
                output_ids = self._model.generate(
                    **encodings,
                    generation_config=generation_config,
                )

            # Slice off the input tokens so we only decode the new tokens.
            for i, (out_ids, in_len) in enumerate(zip(output_ids, input_lengths)):
                new_token_ids = out_ids[in_len:]
                raw_text = self._tokenizer.decode(
                    new_token_ids, skip_special_tokens=True
                ).strip()
                predicted = extract_answer(raw_text)
                results.append(
                    GenerationResult(
                        raw_output=raw_text,
                        predicted_answer=predicted,
                        input_tokens=int(in_len),
                        output_tokens=len(new_token_ids),
                    )
                )

        return results

    def format_prompt(
        self,
        system_message: str,
        user_message: str,
    ) -> str:
        """
        Build a ChatML-formatted string for Qwen2.5-Instruct.

        Qwen2.5 uses the standard ChatML template:
            <|im_start|>system\\n{system}<|im_end|>
            <|im_start|>user\\n{user}<|im_end|>
            <|im_start|>assistant\\n

        We use apply_chat_template() so the template stays in sync with
        whatever the tokenizer's config specifies.
        """
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        # add_generation_prompt=True appends the assistant turn opener so the
        # model continues from there rather than predicting a new role token.
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in `text` according to the Qwen tokenizer."""
        if self._tokenizer is None:
            raise RuntimeError("Call load() before count_tokens().")
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    @property
    def is_loaded(self) -> bool:
        return self._model is not None


# ------------------------------------------------------------------
# Answer extraction (module-level for testability)
# ------------------------------------------------------------------


def extract_answer(text: str) -> Optional[str]:
    """
    Extract a single A/B/C/D answer letter from a model's free-form output.

    Tries patterns from most specific to least specific. Returns the uppercase
    letter, or None if no answer can be reliably extracted.

    Examples handled:
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
    Format a choices dictionary into the multi-line block shown in prompts.

    Input:  {"A": "Hà Nội", "B": "Đà Nẵng", "C": "Huế", "D": "Hội An"}
    Output: "A. Hà Nội\\nB. Đà Nẵng\\nC. Huế\\nD. Hội An"
    """
    lines = []
    for letter in ["A", "B", "C", "D"]:
        if letter in choices:
            lines.append(f"{letter}. {choices[letter]}")
    return "\n".join(lines)
