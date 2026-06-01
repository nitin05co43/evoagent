"""
data.py — Load ViMMRC data for EvoAgent.

Loads from the bundled data/ directory (no internet, no accounts needed).
The data/ folder contains synthetic Vietnamese MCQ data with the same
structure as ViMMRC 2.0. Replace with real ViMMRC data if you have access.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default: bundled data next to this file
_DEFAULT_DATA_DIR = Path(__file__).parent / "data"


def _load_json_split(path: Path) -> list[dict]:
    """
    Parse a ViMMRC-format JSON file into a flat list of examples.

    Each example has: id, context, question, options (list of 4), answer (A/B/C/D).
    """
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    examples = []
    for article in raw.get("data", []):
        for para in article.get("paragraphs", []):
            context = para.get("context", "")
            for qa in para.get("qas", []):
                options = qa.get("options", [])
                # Normalise option text: strip leading "A. " etc if present
                cleaned = []
                for opt in options:
                    opt = opt.strip()
                    if len(opt) >= 3 and opt[1] == "." and opt[0] in "ABCD":
                        opt = opt[2:].strip()
                    cleaned.append(opt)

                answer = qa.get("answer", "A").strip().upper()
                if answer not in {"A", "B", "C", "D"}:
                    answer = "A"

                examples.append({
                    "id": qa.get("id", str(len(examples))),
                    "context": context,
                    "question": qa.get("question", ""),
                    "options": cleaned,
                    "answer": answer,
                    "question_type": qa.get("question_type", ""),
                })
    return examples


def load_vimmrc(data_dir: Optional[Path] = None) -> dict:
    """
    Load ViMMRC splits from local JSON files.

    Returns a HuggingFace DatasetDict with train / validation / test splits.
    """
    from datasets import Dataset, DatasetDict

    data_dir = Path(data_dir or _DEFAULT_DATA_DIR)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            "Run: python make_synthetic_data.py"
        )

    splits = {}
    file_map = {
        "train": data_dir / "train.json",
        "validation": data_dir / "dev.json",
        "test": data_dir / "test.json",
    }

    for split, path in file_map.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing data file: {path}\n"
                "Run: python make_synthetic_data.py"
            )
        examples = _load_json_split(path)
        splits[split] = Dataset.from_list(examples)
        logger.info("Loaded %d examples for %s split.", len(examples), split)

    return DatasetDict(splits)


def load_vimmrc_splits(
    train_size: int = 100,
    dev_size: Optional[int] = None,
    seed: int = 42,
    data_dir: Optional[Path] = None,
):
    """
    Load ViMMRC and return (train_subset, dev_split) ready for EvoAgent.

    Parameters
    ----------
    train_size: number of train examples for cheap mid-loop eval.
    dev_size: number of dev examples (None = full dev split).
    seed: random seed for shuffling.
    data_dir: path to directory containing train.json / dev.json / test.json.
    """
    ds = load_vimmrc(data_dir=data_dir)

    train_split = ds["train"].shuffle(seed=seed)
    train_subset = train_split.select(range(min(train_size, len(train_split))))

    dev_split = ds["validation"].shuffle(seed=seed)
    if dev_size is not None:
        dev_split = dev_split.select(range(min(dev_size, len(dev_split))))

    logger.info(
        "Splits ready: train_subset=%d, dev=%d.",
        len(train_subset), len(dev_split),
    )
    return train_subset, dev_split
