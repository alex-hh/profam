import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline

csv_dir = "proteingym_variants/20250722_113929"

csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))

# Generate a distinct color for each CSV using a continuous colormap
colors = plt.cm.rainbow(np.linspace(0, 1, len(csv_files)))

# Containers to reuse data in the subplot section
all_likelihood_vals = []  # list of np.ndarray
all_spearman_norm_vals = []  # list of np.ndarray
all_n_prompt_vals = []  # list of np.ndarray

highest_likelihood_vals = []
last_value_vals = []
mean_vals = []
upper_quartile_vals = []
upper_quartile_exclude_lt1_vals = []
highest_with_exclusion_vals = []
exclusion_threshold = -1.7
for i, csv_file in enumerate(csv_files):
    df = pd.read_csv(csv_file)
    df = df.sort_values(by='n_prompt_seqs', ascending=True)
    last_value_vals.append(df.iloc[-1]['spearman'])
    df = df.sort_values(by='mean_log_likelihood', ascending=True)
    likelihood_vals = df['mean_log_likelihood'].values
    spearman_vals = df['spearman'].values
    n_prompt_vals = df['n_prompt_seqs'].values

    # Normalise spearman values to the range [0, 1] for this CSV
    spearman_min, spearman_max = spearman_vals.min(), spearman_vals.max()
    # Avoid division by zero if all values are identical
    if spearman_max - spearman_min != 0:
        spearman_norm = (spearman_vals - spearman_min) / (spearman_max - spearman_min)
    else:
        spearman_norm = np.zeros_like(spearman_vals)

    # Scatter plot for this CSV (no legend, semi-transparent points)
    plt.scatter(likelihood_vals, spearman_norm, color=colors[i], alpha=0.3, s=10)

    # Collect arrays for later subplot visualization
    all_likelihood_vals.append(likelihood_vals)
    all_spearman_norm_vals.append(spearman_norm)
    all_n_prompt_vals.append(n_prompt_vals)

    highest_likelihood_vals.append(spearman_vals[-1])
    mean_vals.append(np.mean(spearman_vals))
    upper_quartile = np.percentile(likelihood_vals, 75)
    upper_quartile_vals.append(df[df['mean_log_likelihood'] >= upper_quartile]['spearman'].values.mean())
    exclude_lt1_vals = df[df['mean_log_likelihood'] <= max(exclusion_threshold, likelihood_vals.min())]
    upper_quartile_exclude = np.percentile(exclude_lt1_vals['mean_log_likelihood'], 75)
    upper_quartile_exclude_lt1_vals.append(exclude_lt1_vals[exclude_lt1_vals['mean_log_likelihood'] >= upper_quartile_exclude]['spearman'].values.mean())
    highest_with_exclusion_vals.append(exclude_lt1_vals.iloc[-1]['spearman'])

# print the mean of the lists with labels
print(f"Highest likelihood spearman: {np.mean(highest_likelihood_vals)}")
print(f"Last value spearman: {np.mean(last_value_vals)}")
print(f"Mean spearman: {np.mean(mean_vals)}")
print(f"Upper quartile spearman: {np.mean(upper_quartile_vals)}")
print(f"Upper quartile exclude lt1 spearman: {np.mean(upper_quartile_exclude_lt1_vals)}")
print(f"Highest with exclusion spearman: {np.mean(highest_with_exclusion_vals)}")

# -----------------------------------------------------------------------------
# Combined spline fitted over all CSVs (black line)
# -----------------------------------------------------------------------------

# Concatenate all stored likelihood and spearman values
combined_x = np.concatenate(all_likelihood_vals)
combined_y = np.concatenate(all_spearman_norm_vals)

# Sort for spline fitting (x must be increasing)
sort_idx_global = np.argsort(combined_x)
xs_g, ys_g = combined_x[sort_idx_global], combined_y[sort_idx_global]

# Determine internal knots (reuse logic from subplots)
unique_x_g = np.unique(xs_g)
internal_knots_g = []
if len(unique_x_g) > 15:
    internal_knots_g = np.quantile(unique_x_g, [0.1, 0.2, 0.4, 0.8, 0.9]).tolist()
elif len(unique_x_g) > 7:
    internal_knots_g = [np.median(unique_x_g)]

try:
    if len(internal_knots_g) > 0:
        global_spline = LSQUnivariateSpline(xs_g, ys_g, t=internal_knots_g, k=2)
    else:
        global_spline = UnivariateSpline(xs_g, ys_g, k=3, s=0)

    x_global_smooth = np.linspace(xs_g.min(), xs_g.max(), 400)
    y_global_smooth = global_spline(x_global_smooth)
    plt.plot(x_global_smooth, y_global_smooth, color="black", linewidth=2)
except Exception as e:
    print(f"Global spline fitting failed: {e}")

# Final plot aesthetics
plt.xlabel("Mean log likelihood")
plt.ylabel("Normalised spearman")
plt.title("Spearman vs Likelihood across CSVs")
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(f"{csv_dir}/spearman_vs_likelihood.png")
plt.show()

# -----------------------------------------------------------------------------
# Combined scatter & spline: n_prompt_seqs on x-axis
# -----------------------------------------------------------------------------

plt.figure(figsize=(8, 6))

# Scatter points for each CSV
for i in range(len(csv_files)):
    plt.scatter(all_n_prompt_vals[i], all_spearman_norm_vals[i], color=colors[i], alpha=0.3, s=10)

# Fit global spline across all points (limited knots)
combined_x_np = np.concatenate(all_n_prompt_vals)
combined_y_np = np.concatenate(all_spearman_norm_vals)

sort_idx_np = np.argsort(combined_x_np)
xs_np, ys_np = combined_x_np[sort_idx_np], combined_y_np[sort_idx_np]

unique_x_np = np.unique(xs_np)
internal_knots_np = []
if len(unique_x_np) > 15:
    internal_knots_np = np.quantile(unique_x_np, [0.1, 0.2, 0.4, 0.8, 0.9]).tolist()
elif len(unique_x_np) > 7:
    internal_knots_np = [np.median(unique_x_np)]

try:
    if len(internal_knots_np) > 0:
        spline_np = LSQUnivariateSpline(xs_np, ys_np, t=internal_knots_np, k=2)
    else:
        spline_np = UnivariateSpline(xs_np, ys_np, k=3, s=0)

    x_np_smooth = np.linspace(xs_np.min(), xs_np.max(), 400)
    y_np_smooth = spline_np(x_np_smooth)
    plt.plot(x_np_smooth, y_np_smooth, color="black", linewidth=2)
except Exception as e:
    print(f"Global n_prompt spline fitting failed: {e}")

plt.xlabel("n_prompt_seqs")
plt.ylabel("Normalised spearman")
plt.title("Spearman vs n_prompt_seqs across CSVs")
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(f"{csv_dir}/spearman_vs_n_prompt.png")
plt.show()

# -----------------------------------------------------------------------------
# Additional subplot figure: each CSV in its own subplot (10 rows deep)
# -----------------------------------------------------------------------------

n_rows = 10
n_cols = int(np.ceil(len(csv_files) / n_rows))

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(4 * n_cols, 3 * n_rows),
    sharex=True,  # ensure all subplots share the same x-axis
    squeeze=False,
)
axes_flat = axes.flatten()

# Determine global x-axis limits so every subplot has identical range
global_x_min = min(arr.min() for arr in all_likelihood_vals)
global_x_max = max(arr.max() for arr in all_likelihood_vals)

for i in range(n_rows * n_cols):
    if i >= len(csv_files):
        # Hide unused axes
        axes_flat[i].axis("off")
        continue

    x = all_likelihood_vals[i]
    y = all_spearman_norm_vals[i]

    ax = axes_flat[i]
    # Scatter points in purple
    ax.scatter(x, y, color="purple", alpha=0.3, s=10)

    # Fit a spline with at most 2 internal knots to reduce wiggling
    try:
        # Sort to ensure strictly increasing x for spline routines
        sort_idx = np.argsort(x)
        xs, ys = x[sort_idx], y[sort_idx]

        # Determine internal knot positions (≤2) based on data availability
        unique_x = np.unique(xs)
        internal_knots = []
        if len(unique_x) > 12:
            # Use two interior knots at 33% and 67% quantiles
            internal_knots = np.quantile(unique_x, [0.1, 0.2, 0.4, 0.8, 0.9]).tolist()
        elif len(unique_x) > 6:
            # Use a single interior knot at the median
            internal_knots = [np.median(unique_x)]

        if len(internal_knots) > 0:
            spline = LSQUnivariateSpline(xs, ys, t=internal_knots, k=2)
        else:
            # Fall back to simple cubic spline without interior knots
            spline = UnivariateSpline(xs, ys, k=3, s=0)

        x_smooth = np.linspace(xs.min(), xs.max(), 200)
        y_smooth = spline(x_smooth)
        # Plot spline in blue
        ax.plot(x_smooth, y_smooth, color="blue")

        # Horizontal line at spline maximum
        y_max = y_smooth.max()
        ax.axhline(y_max, linestyle="--", color="grey", alpha=0.5)

        # Vertical dashed line at exclusion_threshold
        ax.axvline(exclusion_threshold, linestyle="--", color="grey", alpha=0.5)
    except Exception as e:
        # In case spline fitting fails (e.g., not enough unique points), skip
        print(f"Spline fitting failed for CSV index {i}: {e}")

    # Minimal axis ticks to keep the plot clean
    ax.tick_params(axis='both', which='major', labelsize=6)

# Apply global x-limits to all populated axes (sharex already links, but set explicitly)
for ax in axes_flat[: len(csv_files)]:
    ax.set_xlim(global_x_min, global_x_max)

# Global figure adjustments and save
fig.tight_layout()
subplot_path = f"{csv_dir}/spearman_vs_likelihood_subplots.png"
fig.savefig(subplot_path)
plt.show()

# -----------------------------------------------------------------------------
# Subplots with n_prompt_seqs on x-axis (10 rows deep)
# -----------------------------------------------------------------------------

fig_np, axes_np = plt.subplots(
    n_rows,
    n_cols,
    figsize=(4 * n_cols, 3 * n_rows),
    sharex=True,
    squeeze=False,
)

axes_np_flat = axes_np.flatten()

# Determine global x-axis limits for n_prompt values
global_x_min_np = min(arr.min() for arr in all_n_prompt_vals)
global_x_max_np = max(arr.max() for arr in all_n_prompt_vals)

for i in range(n_rows * n_cols):
    if i >= len(csv_files):
        axes_np_flat[i].axis("off")
        continue

    x_np = all_n_prompt_vals[i]
    y_np = all_spearman_norm_vals[i]

    ax_np = axes_np_flat[i]
    ax_np.scatter(x_np, y_np, color="purple", alpha=0.3, s=10)

    # Fit spline with ≤2 internal knots
    try:
        sort_idx_local = np.argsort(x_np)
        xs_l, ys_l = x_np[sort_idx_local], y_np[sort_idx_local]

        unique_x_l = np.unique(xs_l)
        int_knots_l = []
        if len(unique_x_l) > 12:
            int_knots_l = np.quantile(unique_x_l, [0.1, 0.2, 0.4, 0.8, 0.9]).tolist()
        elif len(unique_x_l) > 6:
            int_knots_l = [np.median(unique_x_l)]

        if len(int_knots_l) > 0:
            spline_l = LSQUnivariateSpline(xs_l, ys_l, t=int_knots_l, k=2)
        else:
            spline_l = UnivariateSpline(xs_l, ys_l, k=3, s=0)

        xs_smooth = np.linspace(xs_l.min(), xs_l.max(), 200)
        ys_smooth = spline_l(xs_smooth)
        ax_np.plot(xs_smooth, ys_smooth, color="blue")

        # Horizontal line at spline maximum
        y_max_l = ys_smooth.max()
        ax_np.axhline(y_max_l, linestyle="--", color="grey", alpha=0.5)
    except Exception as e:
        print(f"Spline (n_prompt) failed for CSV index {i}: {e}")

    ax_np.tick_params(axis='both', which='major', labelsize=6)

# Apply global x-limits
for ax_np in axes_np_flat[: len(csv_files)]:
    ax_np.set_xlim(global_x_min_np, global_x_max_np)

fig_np.tight_layout()
subplot_path_np = f"{csv_dir}/spearman_vs_n_prompt_subplots.png"
fig_np.savefig(subplot_path_np)
plt.show()




