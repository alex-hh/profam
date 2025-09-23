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
import pandas as pd

indels_csv = "../data/ProteinGym/Summary_performance_DMS_indels_Spearman.csv"
substitutions_csv = "../data/ProteinGym/Summary_performance_DMS_substitutions_Spearman.csv"

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


def _make_horizontal_lineplot(
    csv_path: str,
    output_dir: str,
    out_basename: str,
    title: str,
) -> str:
    """Create a horizontal line plot from a ProteinGym summary CSV.

    Expects columns `Model_name` and `Average_Spearman`.
    One horizontal line per model from x=0 to x=Average_Spearman.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    if not {"Model_name", "Average_Spearman"}.issubset(df.columns):
        raise ValueError(
            f"CSV {csv_path} must contain 'Model_name' and 'Average_Spearman' columns"
        )

    df = (
        df[["Model_name", "Average_Spearman"]]
        .dropna()
        .sort_values("Average_Spearman", ascending=False)
    )

    model_names: List[str] = df["Model_name"].astype(str).tolist()
    scores: List[float] = df["Average_Spearman"].astype(float).tolist()

    num_models = len(model_names)
    if num_models == 0:
        raise ValueError(f"No rows to plot in {csv_path}")

    # Use a Times-like serif font and modest sizes suitable for ~46 rows
    base_font = 12
    with plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": base_font,
            "axes.titlesize": base_font + 2,
            "axes.labelsize": base_font,
            "xtick.labelsize": base_font - 1,
            "ytick.labelsize": base_font - 1,
            "legend.fontsize": base_font,
        }
    ):
        # Height scaled so horizontal lines are about as tall as the text height
        row_height_inches = (base_font / 72.0) * 1.25
        fig_height = max(6.0, num_models * row_height_inches + 1.0)

        # Width scaled by the longest label to avoid clipping
        longest_label = max((len(n) for n in model_names), default=20)
        fig_width = max(12.0, 0.18 * longest_label + 8.0)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=False)

        y_positions = list(range(num_models))
        line_width = max(1.0, base_font / 6.0)  # narrow visual weight, similar to text

        for y, s in enumerate(scores):
            name = model_names[y]
            is_profam = "ProFam" in name
            color = "#D62728" if is_profam else "#4C78A8"  # red for ProFam
            ax.hlines(y, 0.0, s, colors=color, linewidth=line_width)
            ax.plot(s, y, "o", color=color, markersize=max(2.0, base_font / 2.5))

        ax.set_yticks(y_positions)
        ax.set_yticklabels(model_names)
        # Emphasize our model label
        for tick_label, name in zip(ax.get_yticklabels(), model_names):
            if "ProFam" in name:
                tick_label.set_fontweight("bold")
        ax.invert_yaxis()  # Highest score at the top
        ax.set_xlabel("Average Spearman")
        ax.set_title(title)
        ax.set_xlim(0.0, max(scores) * 1.05)
        ax.grid(axis="x", linestyle="--", alpha=0.3)

        # Extra left margin for long labels
        fig.subplots_adjust(left=0.42, right=0.98, top=0.94, bottom=0.06)

        out_png = os.path.join(output_dir, f"{out_basename}.png")
        out_pdf = os.path.join(output_dir, f"{out_basename}.pdf")
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_png


def make_horizontal_lineplots_from_csvs(
    substitutions_csv_path: str,
    indels_csv_path: str,
    output_dir: str,
) -> Tuple[str, str]:
    """Convenience wrapper to render horizontal line plots for both CSVs.

    Returns the output PNG paths for (substitutions, indels).
    """
    sub_out = _make_horizontal_lineplot(
        substitutions_csv_path,
        output_dir,
        out_basename="gym_performance_horizontal_substitutions",
        title="ProteinGym DMS Substitutions",
    )
    indel_out = _make_horizontal_lineplot(
        indels_csv_path,
        output_dir,
        out_basename="gym_performance_horizontal_indels",
        title="ProteinGym DMS INDELs",
    )
    return sub_out, indel_out


if __name__ == "__main__":
    make_horizontal_lineplots_from_csvs(substitutions_csv, indels_csv, save_dir)
    make_barplot(results, save_dir)
    