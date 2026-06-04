"""
analysis.py — Post-hoc analysis and visualisation for EvoAgent results.

Three public functions:
  - plot_learning_curve(history):   accuracy vs. iteration line plot.
  - compute_strategy_diversity(history): pairwise embedding distance matrix.
  - failure_mode_report(history):   per-type accuracy trends + a text report.

All plots are saved to PDF. All functions accept an optional output_dir;
if omitted they save to the current working directory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# Matplotlib is imported lazily inside each function so the module can be
# imported in headless environments without triggering a display backend error.


def plot_learning_curve(
    history,  # StrategyHistory — avoid circular import
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Plot dev (and train) accuracy across EvoAgent iterations.

    Saves a PDF to output_dir/learning_curve.pdf and returns the path.
    Each point corresponds to one evaluated strategy. The best strategy is
    highlighted with a star marker.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir or ".")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = history.summary_table()
    iterations = [r["iteration"] for r in rows]
    dev_accs = [r["dev_accuracy"] for r in rows]
    train_accs = [r["train_accuracy"] for r in rows]

    # Filter out un-evaluated rows.
    scored = [(it, dev, tr) for it, dev, tr in zip(iterations, dev_accs, train_accs) if dev is not None]
    if not scored:
        logger.warning("No evaluated strategies to plot.")
        return output_dir / "learning_curve.pdf"

    iters, devs, trains = zip(*scored)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iters, devs, marker="o", label="Dev accuracy", linewidth=2)
    if any(t is not None for t in trains):
        valid_trains = [(i, t) for i, t in zip(iters, trains) if t is not None]
        ti, tv = zip(*valid_trains)
        ax.plot(ti, tv, marker="s", linestyle="--", label="Train-subset accuracy", linewidth=1.5, alpha=0.7)

    # Star the best point.
    best_idx = int(np.argmax(devs))
    ax.scatter([iters[best_idx]], [devs[best_idx]], marker="*", s=200, color="gold", zorder=5, label=f"Best ({devs[best_idx]:.3f})")

    # Draw baseline reference line at 0.52.
    ax.axhline(0.52, linestyle=":", color="gray", linewidth=1, label="Baseline (~0.52)")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Accuracy")
    ax.set_title("EvoAgent Learning Curve — ViMMRC 2.0")
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)

    path = output_dir / "learning_curve.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Learning curve saved to %s", path)
    return path


def compute_strategy_diversity(
    history,  # StrategyHistory
    output_dir: Optional[Path] = None,
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
) -> tuple[np.ndarray, Path]:
    """
    Compute pairwise cosine similarity between strategy prompt templates.

    Uses sentence-transformers with a multilingual model so Vietnamese text
    in the templates is handled correctly.

    Returns (similarity_matrix, pdf_path).
    The similarity matrix is (N x N) float32 where N = number of strategies.
    The heatmap is saved to output_dir/strategy_diversity.pdf.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error("sentence-transformers is not installed. Run: pip install sentence-transformers")
        raise

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir or ".")
    output_dir.mkdir(parents=True, exist_ok=True)

    templates = [s.prompt_template for s in history.strategies]
    if not templates:
        logger.warning("No strategies to compute diversity for.")
        empty = np.zeros((0, 0), dtype=np.float32)
        return empty, output_dir / "strategy_diversity.pdf"

    logger.info("Loading sentence transformer '%s'…", model_name)
    encoder = SentenceTransformer(model_name)
    embeddings = encoder.encode(templates, normalize_embeddings=True)  # shape: (N, D)

    # Cosine similarity = dot product of unit vectors.
    sim_matrix = np.dot(embeddings, embeddings.T)

    labels = [f"Iter {s.metadata.iteration}" for s in history.strategies]

    fig, ax = plt.subplots(figsize=(max(4, len(labels)), max(3, len(labels))))
    im = ax.imshow(sim_matrix, vmin=0, vmax=1, cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="Cosine similarity")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title("Strategy Prompt Template Similarity")

    # Annotate cells.
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{sim_matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)

    path = output_dir / "strategy_diversity.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Strategy diversity heatmap saved to %s", path)
    return sim_matrix, path


def failure_mode_report(
    history,  # StrategyHistory
    output_dir: Optional[Path] = None,
) -> tuple[str, Path]:
    """
    Generate a failure-mode report: per-type accuracy trends and per-iteration breakdown.

    Returns (report_text, pdf_path). Also writes a .txt report.
    The PDF shows per-type accuracy vs. iteration for each question type.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir or ".")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all question types seen across all reflections.
    all_types: set[str] = set()
    type_trends: dict[str, list[tuple[int, float]]] = {}

    for strategy, reflection in zip(history.strategies, history.reflections):
        if reflection is None:
            continue
        iteration = strategy.metadata.iteration
        for q_type, acc in reflection.accuracy_by_type.items():
            all_types.add(q_type)
            if q_type not in type_trends:
                type_trends[q_type] = []
            type_trends[q_type].append((iteration, acc))

    # Build text report.
    lines = ["=" * 60, "EvoAgent Failure Mode Report", "=" * 60, ""]

    if not history.strategies:
        lines.append("No strategies in history.")
    else:
        best = history.best_strategy()
        lines.append(
            f"Total iterations: {len(history.strategies)}\n"
            f"Best strategy: iteration {best.metadata.iteration if best else '—'}, "
            "dev accuracy {}\n".format('{:.3f}'.format(best.metadata.dev_accuracy) if best and best.metadata.dev_accuracy else '—')
        )

        lines.append("Per-type accuracy (from reflections):")
        for q_type in sorted(all_types):
            trend = type_trends.get(q_type, [])
            if trend:
                accs = [a for _, a in trend]
                lines.append(
                    f"  {q_type}: min={min(accs):.3f}, max={max(accs):.3f}, "
                    f"final={accs[-1]:.3f}, trend={'↑' if len(accs) > 1 and accs[-1] > accs[0] else '↓' if len(accs) > 1 and accs[-1] < accs[0] else '→'}"
                )

        lines.append("\nHypotheses by iteration:")
        for strategy, reflection in zip(history.strategies, history.reflections):
            if reflection:
                lines.append(
                    f"  Iter {strategy.metadata.iteration}: {reflection.hypothesis[:200]}"
                )

    report_text = "\n".join(lines)

    txt_path = output_dir / "failure_mode_report.txt"
    txt_path.write_text(report_text, encoding="utf-8")

    # Build per-type accuracy trend plot.
    pdf_path = output_dir / "failure_mode_report.pdf"
    if type_trends:
        n_types = len(type_trends)
        fig, axes = plt.subplots(
            nrows=(n_types + 1) // 2,
            ncols=2,
            figsize=(12, 3 * ((n_types + 1) // 2)),
            squeeze=False,
        )
        axes_flat = [ax for row in axes for ax in row]

        for ax, (q_type, trend) in zip(axes_flat, sorted(type_trends.items())):
            iters, accs = zip(*sorted(trend))
            ax.plot(iters, accs, marker="o")
            ax.axhline(0.52, linestyle=":", color="gray", linewidth=1)
            ax.set_title(q_type)
            ax.set_ylabel("Accuracy")
            ax.set_xlabel("Iteration")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)

        # Hide empty subplots.
        for ax in axes_flat[n_types:]:
            ax.set_visible(False)

        fig.suptitle("Per-Type Accuracy Across Iterations", y=1.02)
        fig.tight_layout()
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Failure mode report (PDF) saved to %s", pdf_path)
    else:
        logger.info("No reflection data available for per-type trend plot.")

    logger.info("Failure mode report (text) saved to %s", txt_path)
    return report_text, pdf_path
