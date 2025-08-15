import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score

csv_path = "/Users/judewells/Downloads/SING_RAND2/*.csv"
dfs = [pd.read_csv(f) for f in glob.glob(csv_path)]
df = pd.concat(dfs)

"""
'family_id', 'sample_plddt', 'sample_lens',
       'num_samples_greater_than_max_length', 'prompt_plddt', 'prompt_lens',
       'min_tm_score', 'max_tm_score', 'mean_tm_score', 'mean_sample_length',
       'max_sample_length', 'min_sample_length', 'mean_prompt_length',
       'max_prompt_length', 'min_prompt_length', 'n_seqs_in_prompt',
       'min_sample_prompt_identity', 'max_sample_prompt_identity',
       'mean_sample_prompt_identity'

create plots with r^2 reported for max_sample_prompt_identity vs max_tm_score
and max_sample_prompt_identity vs sample_plddt
and prompt_plddt vs sample_plddt
"""
out_dir = "/Users/judewells/Downloads/v8_plots"

# Ensure output directory exists
os.makedirs(out_dir, exist_ok=True)

# Helper function to create scatter plot with R^2


def create_scatter_with_r2(
    df, x_col, y_col, xlabel=None, ylabel=None, title=None, fname=None
):
    """Generate a scatter (reg) plot with R^2 annotation and save it.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing the variables.
    x_col : str
        Column name for x-axis.
    y_col : str
        Column name for y-axis.
    xlabel : str, optional
        Label for x-axis. Defaults to x_col.
    ylabel : str, optional
        Label for y-axis. Defaults to y_col.
    title : str, optional
        Plot title. Defaults to "{xlabel} vs {ylabel}".
    fname : str, optional
        File name to write (inside out_dir). If not provided, generated from x and y names.
    """
    # Drop rows with missing values in either column
    data = df[[x_col, y_col]].dropna()
    x = data[x_col]
    y = data[y_col]

    # Calculate R^2
    if len(data) > 1:
        r2 = np.corrcoef(x, y)[0, 1] ** 2
    else:
        r2 = np.nan

    # Plot
    plt.figure(figsize=(6, 5))
    sns.regplot(
        x=x, y=y, scatter_kws={"s": 10, "alpha": 0.1}, line_kws={"color": "red"}
    )

    xlabel = xlabel or x_col
    ylabel = ylabel or y_col
    title = title or f"{xlabel} vs {ylabel}"

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Position R^2 text in upper left corner of the plot
    x_text = x.min() + 0.05 * (x.max() - x.min())
    y_text = y.max() - 0.05 * (y.max() - y.min())
    plt.text(x_text, y_text, f"$R^2$ = {r2:.3f}", fontsize=12, verticalalignment="top")

    plt.tight_layout()

    if fname is None:
        safe_x = x_col.replace(" ", "_")
        safe_y = y_col.replace(" ", "_")
        fname = f"{safe_x}_vs_{safe_y}.png"

    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=300)
    plt.close()


# NEW FUNCTION FOR PAIRED PLOTS


def create_dual_scatter_with_r2(
    df, x_col, y_col, xlabel=None, ylabel=None, title=None, fname=None
):
    """Scatter+reg plots for ProFam and `mut_` data on the same axes.

    Two sets of points/lines are drawn in different colors and the R² for each
    is annotated in the top-left corner of the figure.
    """

    # Build names for random_mutation columns
    mut_x_col = f"mut_{x_col}"
    mut_y_col = f"mut_{y_col}"

    # Drop NA rows for each set separately
    data_orig = df[[x_col, y_col]].dropna()
    data_mut = df[[mut_x_col, mut_y_col]].dropna()

    # Compute R² values (if at least 2 points)
    def _r2(a, b):
        if len(a) > 1:
            return np.corrcoef(a, b)[0, 1] ** 2
        return np.nan

    r2_orig = _r2(data_orig[x_col], data_orig[y_col])
    r2_mut = _r2(data_mut[mut_x_col], data_mut[mut_y_col])

    # Plot
    plt.figure(figsize=(6, 5))

    sns.regplot(
        x=data_orig[x_col],
        y=data_orig[y_col],
        scatter_kws={"s": 10, "alpha": 0.3, "color": "C0"},
        line_kws={"color": "C0"},
        label="ProFam",
    )

    sns.regplot(
        x=data_mut[mut_x_col],
        y=data_mut[mut_y_col],
        scatter_kws={"s": 10, "alpha": 0.3, "color": "C1"},
        line_kws={"color": "C1"},
        label="random_mutation",
    )

    xlabel = xlabel or x_col
    ylabel = ylabel or y_col
    title = title or f"{xlabel} vs {ylabel}"

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Axis limits for text placement
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    x_range = x_max - x_min
    y_range = y_max - y_min

    plt.text(
        x_min + 0.05 * x_range,
        y_max - 0.05 * y_range,
        f"orig $R^2$ = {r2_orig:.3f}",
        color="C0",
        fontsize=11,
        verticalalignment="top",
    )
    plt.text(
        x_min + 0.05 * x_range,
        y_max - 0.15 * y_range,
        f"mut  $R^2$ = {r2_mut:.3f}",
        color="C1",
        fontsize=11,
        verticalalignment="top",
    )

    plt.legend()
    plt.tight_layout()

    # File name
    if fname is None:
        safe_x = x_col.replace(" ", "_")
        safe_y = y_col.replace(" ", "_")
        fname = f"{safe_x}_vs_{safe_y}.png"

    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=300)
    plt.close()


# Define pairs to plot: (x_col, y_col, xlabel, ylabel, title, filename)
plot_specs = [
    (
        "max_sample_prompt_identity",
        "max_tm_score",
        "Max Sample–Prompt Identity",
        "Max TM-score",
        None,
        "max_identity_vs_max_tm.png",
    ),
    (
        "max_sample_prompt_identity",
        "sample_plddt",
        "Max Sample–Prompt Identity",
        "Sample pLDDT",
        None,
        "max_identity_vs_sample_plddt.png",
    ),
    (
        "prompt_plddt",
        "sample_plddt",
        "Prompt pLDDT",
        "Sample pLDDT",
        None,
        "prompt_plddt_vs_sample_plddt.png",
    ),
]

for x_col, y_col, xlabel, ylabel, title, fname in plot_specs:
    create_dual_scatter_with_r2(df, x_col, y_col, xlabel, ylabel, title, fname)

print(f"Saved plots to {out_dir}")
