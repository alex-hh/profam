results = {
    "ProFam": 0.472,
    "PoET": 0.47,
    "MSA Pairformer": 0.47,
    "ESM3 open (1.4B)": 0.466,
    "VespaG": 0.458,
    "MSA Transformer": 0.432,
    "ESM2 (650m)": 0.414,
    "AIDO Protein-RAG (16B)": 0.518,
    "VenusREM": 0.518,
    "RITA XL": 0.373,
    "Progen3 (3B)": 0.392,
    "xTrimo-10B-MLM": 0.396,
}
save_dir = "plots_for_paper/gym_performance_barplot"
import os
from typing import List, Tuple

import matplotlib.pyplot as plt

# Make all text roughly 2x default size
plt.rcParams.update({
    "font.size": 20,           # base text size (default ~10)
    "axes.titlesize": 24,     # axes title
    "axes.labelsize": 20,     # x/y labels
    "xtick.labelsize": 20,    # x tick labels
    "ytick.labelsize": 20,    # y tick labels
    "legend.fontsize": 20,    # legend text
    "figure.titlesize": 28,   # figure-level title if used
})


def make_barplot(model_results: dict, output_dir: str) -> str:
    # Sort by score (descending)
    sorted_items: List[Tuple[str, float]] = sorted(
        model_results.items(), key=lambda kv: kv[1], reverse=True
    )
    model_names: List[str] = [k for k, _ in sorted_items]
    scores: List[float] = [v for _, v in sorted_items]

    os.makedirs(output_dir, exist_ok=True)

    # Dynamic figure width based on number of bars; ensure enough space for labels
    num_models = len(model_names)
    fig_width = max(12, 0.85 * num_models + 4)
    fig, ax = plt.subplots(figsize=(fig_width, 6), constrained_layout=False)

    ax.bar(range(num_models), scores, color="#4C78A8")
    ax.set_ylabel("Average Spearman")
    ax.set_xticks(range(num_models))
    ax.set_xticklabels(model_names, rotation=60, ha="right", rotation_mode="anchor")
    ax.set_ylim(0, max(scores) * 1.1)
    ax.set_title("ProteinGym DMS Substitutions Performance Comparison")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Extra bottom margin for long, rotated labels
    fig.subplots_adjust(bottom=0.35, left=0.08, right=0.98)

    out_png = os.path.join(output_dir, "gym_performance_barplot.png")
    out_pdf = os.path.join(output_dir, "gym_performance_barplot.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_png


if __name__ == "__main__":
    make_barplot(results, save_dir)