"""
main.py — CLI entry point for EvoAgent.

Usage:
    python main.py \\
        --T 5 \\
        --output-dir ./runs/experiment_01 \\
        --train-size 100 \\
        --dev-size 200

Run `python main.py --help` for all options.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def setup_logging(output_dir: Path, level: str = "INFO") -> None:
    """Configure root logger to write to both stderr and a log file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Console handler.
    ch = logging.StreamHandler(sys.stderr)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # File handler.
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    root.addHandler(fh)

    logging.info("Logging to %s", log_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EvoAgent: self-improving LLM agent for Vietnamese reading comprehension.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core loop parameters
    parser.add_argument(
        "--T",
        type=int,
        default=5,
        help="Number of EvoAgent iterations (including the seed at iteration 0).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CHECKPOINT_PATH",
        help="Path to an existing history.jsonl file to resume from.",
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="sonlam1102/vimmrc2.0",
        help="HuggingFace dataset ID for ViMMRC 2.0.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=100,
        help="Number of training examples to use for the cheap mid-loop eval.",
    )
    parser.add_argument(
        "--dev-size",
        type=int,
        default=None,
        help="Number of dev examples (None = use full dev split).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset shuffling.",
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model ID or local path for inference.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Inference batch size. Reduce to 4 if you hit VRAM OOM errors.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum new tokens per generation. Increase to 512 for long CoT.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. 0.0 = greedy (recommended for evaluation).",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        default=False,
        help="Disable 4-bit quantization (requires more VRAM; use for testing on GPU with ≥24 GB).",
    )

    # Gemini
    parser.add_argument(
        "--gemini-model",
        type=str,
        default="gemini-2.0-flash",
        help="Gemini model ID for propose and reflect calls.",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./runs/default",
        help="Directory to save history, eval results, and analysis plots.",
    )
    parser.add_argument(
        "--early-stop",
        type=float,
        default=1.0,
        help="Stop early if dev accuracy reaches this threshold.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        default=False,
        help="Skip post-run analysis plots (useful for quick tests).",
    )

    return parser.parse_args()


def load_dataset_splits(
    dataset_id: str,
    train_size: int,
    dev_size: int | None,
    seed: int,
):
    """
    Load ViMMRC 2.0 from HuggingFace and return (train_subset, dev_split).

    The train subset is a small, shuffled slice of the training split used
    for cheap mid-iteration evaluation. The dev split is used for the
    authoritative accuracy numbers.
    """
    import logging
    logger = logging.getLogger(__name__)

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' library is required. Install with: pip install datasets"
        )

    import os
    hf_token = os.environ.get("HF_TOKEN")
    logger.info("Loading dataset '%s' from HuggingFace…", dataset_id)
    ds = load_dataset(dataset_id, token=hf_token)

    train_split = ds["train"].shuffle(seed=seed)
    train_subset = train_split.select(range(min(train_size, len(train_split))))

    dev_split = ds["validation"] if "validation" in ds else ds.get("dev", ds["test"])
    if dev_size is not None:
        dev_split = dev_split.shuffle(seed=seed).select(range(min(dev_size, len(dev_split))))

    logger.info(
        "Dataset loaded: train_subset=%d, dev=%d.",
        len(train_subset),
        len(dev_split),
    )
    return train_subset, dev_split


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    setup_logging(output_dir, args.log_level)

    logger = logging.getLogger(__name__)
    logger.info("EvoAgent starting. Args: %s", vars(args))

    # Save args to output dir for reproducibility.
    (output_dir / "args.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )

    # ----------------------------------------------------------------
    # Load dataset
    # ----------------------------------------------------------------
    train_subset, dev_split = load_dataset_splits(
        dataset_id=args.dataset,
        train_size=args.train_size,
        dev_size=args.dev_size,
        seed=args.seed,
    )

    # ----------------------------------------------------------------
    # Load model
    # ----------------------------------------------------------------
    from model import QwenInference

    logger.info("Initialising QwenInference with model '%s'…", args.model)
    model = QwenInference(
        model_name_or_path=args.model,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        use_4bit=not args.no_4bit,
    )
    model.load()

    # ----------------------------------------------------------------
    # Run EvoAgent
    # ----------------------------------------------------------------
    from harness import run_evoagent

    history = run_evoagent(
        T=args.T,
        train_dataset=train_subset,
        dev_dataset=dev_split,
        model=model,
        output_dir=output_dir,
        resume_from=Path(args.resume) if args.resume else None,
        early_stop_accuracy=args.early_stop,
        gemini_model=args.gemini_model,
    )

    # ----------------------------------------------------------------
    # Post-run analysis
    # ----------------------------------------------------------------
    if not args.skip_analysis:
        from analysis import (
            compute_strategy_diversity,
            failure_mode_report,
            plot_learning_curve,
        )

        logger.info("Running post-run analysis…")
        try:
            plot_learning_curve(history, output_dir=output_dir)
        except Exception as exc:
            logger.error("Learning curve plot failed: %s", exc)

        try:
            compute_strategy_diversity(history, output_dir=output_dir)
        except Exception as exc:
            logger.error("Diversity computation failed: %s", exc)

        try:
            report_text, _ = failure_mode_report(history, output_dir=output_dir)
            print("\n" + report_text)
        except Exception as exc:
            logger.error("Failure mode report failed: %s", exc)

    # ----------------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------------
    best = history.best_strategy()
    if best:
        logger.info(
            "Run complete. Best strategy: iteration=%d, dev_accuracy=%.3f.",
            best.metadata.iteration,
            best.metadata.dev_accuracy,
        )
        print(
            f"\nEvoAgent complete.\n"
            f"Best strategy: iteration {best.metadata.iteration}, "
            f"dev accuracy = {best.metadata.dev_accuracy:.3f}\n"
            f"Results saved to: {output_dir.resolve()}"
        )
    else:
        logger.info("Run complete. No strategies were evaluated.")


if __name__ == "__main__":
    main()
