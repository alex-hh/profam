import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline
from scipy.stats import spearmanr

def get_top_k_spearman_vals(lls, k, dms_scores):
    lls_mean = lls.mean(axis=0)
    lls_mean_sorted_indices = np.argsort(-lls_mean)
    lls_sorted = lls[lls_mean_sorted_indices]
    lls_to_use_sorted = lls_sorted[:k]
    lls_mean_sorted = lls_to_use_sorted.mean(axis=0)
    return spearmanr(lls_mean_sorted, dms_scores)[0]

csv_dir = "proteingym_variants/20250724_224201"
csv_dir = "logs/abyoeovl_openfold_fs50_ur90_memmap_251m_copied_2025-06-23_22-18/20250726_173620"

csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
# randomly sample 50 csv files

np.random.seed(42)
csv_files = np.random.choice(csv_files, size=50, replace=False)
# Generate a distinct color for each CSV using a continuous colormap
colors = plt.cm.rainbow(np.linspace(0, 1, len(csv_files)))

# Containers to reuse data in the subplot section
all_likelihood_vals = []  # list of np.ndarray
all_spearman_norm_vals = []  # list of np.ndarray
all_n_prompt_vals = []  # list of np.ndarray
all_spearman_vals = []  # list of np.ndarray (raw, un-normalised)
# NEW METRIC CONTAINERS
all_likelihood_range_vals = []  # range (max-min) per sequence
all_likelihood_var_vals = []    # variance per sequence
all_likelihood_qcount_vals = [] # unique quantised (2dp) likelihood count per sequence


highest_likelihood_vals = []
last_value_vals = []
mean_vals = []
upper_quartile_vals = []
upper_quartile_exclude_lt1_vals = []
highest_with_exclusion_vals = []
exclusion_threshold = -1.2
per_assay_ensemble_spearman_vals = {}
for i, csv_file in enumerate(csv_files):
    npz_file = csv_file.replace(".csv", "_lls.npz")
    data = np.load(npz_file)
    lls = data["lls"]
    df = pd.read_csv(csv_file)
    df = df.sort_values(by='n_prompt_seqs', ascending=True)
    last_value_vals.append(df.iloc[-1]['spearman'])
    df = df.sort_values(by='mean_log_likelihood', ascending=True)
    likelihood_vals = df['mean_log_likelihood'].values
    spearman_vals = df['spearman'].values
    n_prompt_vals = df['n_prompt_seqs'].values


    # ---------------------------------------------------------------------
    # NEW METRICS (computed row-wise over the lls array)
    # ---------------------------------------------------------------------

    # 1) Full range of log-likelihoods for each sequence (max – min)
    likelihood_ranges = lls.max(axis=1) - lls.min(axis=1)

    # 2) Variance of log-likelihoods for each sequence
    likelihood_variances = lls.var(axis=1)

    # 3) Count of unique quantised (2dp) likelihoods per sequence
    quantised_lls = np.round(lls, 4)
    likelihood_qcounts = np.apply_along_axis(lambda x: len(np.unique(x)), 1, quantised_lls)

    # Re-order metrics to stay aligned with current df ordering (df index
    # corresponds to the original row number prior to sorting)
    likelihood_ranges = likelihood_ranges[df.index]
    likelihood_variances = likelihood_variances[df.index]
    likelihood_qcounts = likelihood_qcounts[df.index]

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
    # Store the raw spearman values (aligned with likelihood_vals)
    all_spearman_vals.append(spearman_vals)

    # ------------------------------------------------------------------
    # Collect new metric arrays for later visualisation
    # ------------------------------------------------------------------

    all_likelihood_range_vals.append(likelihood_ranges)
    all_likelihood_var_vals.append(likelihood_variances)
    all_likelihood_qcount_vals.append(likelihood_qcounts)

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
    # Subtitle = basename of the CSV
    ax.set_title(os.path.basename(csv_files[i]), fontsize=7)

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
    # Subtitle = basename of the CSV
    ax_np.set_title(os.path.basename(csv_files[i]), fontsize=7)

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

# -----------------------------------------------------------------------------
# Scatterplots of n_prompt_seqs vs raw spearman grouped by likelihood bins
# -----------------------------------------------------------------------------

bin_ranges = [
    (-np.inf, -3.5),
    (-3.5, -2.6),
    (-2.6, -2.1),
    (-2.1, -1.8),
    (-1.8, -1.5),
    (-1.5, 0),
]
bin_labels = [
    "(-Inf, -3.5]",
    "(-3.5, -2.6]",
    "(-2.6, -2.1]",
    "(-2.1, -1.8]",
    "(-1.8, -1.5]",
    "(-1.5, 0]",
]

fig_bins, axes_bins = plt.subplots(1, len(bin_ranges), figsize=(5 * len(bin_ranges), 4), sharey=True)

# Ensure axes_bins is iterable even when only one subplot is returned
if len(bin_ranges) == 1:
    axes_bins = [axes_bins]

for b_idx, ((low, high), label) in enumerate(zip(bin_ranges, bin_labels)):
    ax = axes_bins[b_idx]

    # Collect points across all CSVs for this bin to enable spline fitting
    combined_x_bin = []
    combined_y_bin = []

    for i in range(len(csv_files)):
        # Only include points where n_prompt_seqs is between 0 and 35
        mask = (
            (all_likelihood_vals[i] > low)
            & (all_likelihood_vals[i] <= high)
            & (all_n_prompt_vals[i] >= 0)
            & (all_n_prompt_vals[i] <= 35)
        )
        if np.any(mask):
            x_bin = all_n_prompt_vals[i][mask]
            y_bin = all_spearman_norm_vals[i][mask]
            # Scatter individual CSV points in their colour
            ax.scatter(x_bin, y_bin, color=colors[i], alpha=0.4, s=10)

            combined_x_bin.append(x_bin)
            combined_y_bin.append(y_bin)

    # Fit a smoothing spline across all gathered points in this bin, if enough data
    if len(combined_x_bin) > 0:
        xs_bin = np.concatenate(combined_x_bin)
        ys_bin = np.concatenate(combined_y_bin)

        try:
            sort_idx_bin = np.argsort(xs_bin)
            xs_sorted, ys_sorted = xs_bin[sort_idx_bin], ys_bin[sort_idx_bin]

            unique_x_bin = np.unique(xs_sorted)
            internal_knots_bin = []
            if len(unique_x_bin) > 15:
                internal_knots_bin = np.quantile(unique_x_bin, [0.1, 0.2, 0.4, 0.8, 0.9]).tolist()
            elif len(unique_x_bin) > 7:
                internal_knots_bin = [np.median(unique_x_bin)]

            if len(internal_knots_bin) > 0:
                spline_bin = LSQUnivariateSpline(xs_sorted, ys_sorted, t=internal_knots_bin, k=2)
            else:
                spline_bin = UnivariateSpline(xs_sorted, ys_sorted, k=3, s=0)

            xs_smooth_bin = np.linspace(xs_sorted.min(), xs_sorted.max(), 300)
            ys_smooth_bin = spline_bin(xs_smooth_bin)
            ax.plot(xs_smooth_bin, ys_smooth_bin, color="black", linewidth=2)
        except Exception as e:
            # In case fitting fails (e.g., insufficient unique points) skip
            print(f"Spline fitting failed for bin {label}: {e}")
    ax.set_title(label)
    ax.set_xlabel("n_prompt_seqs")
    if b_idx == 0:
        ax.set_ylabel("Spearman")
    ax.grid(True, alpha=0.2)

fig_bins.suptitle("n_prompt_seqs vs Spearman by likelihood bin")
fig_bins.tight_layout(rect=[0, 0.03, 1, 0.95])
fig_bins.savefig(f"{csv_dir}/n_prompt_vs_spearman_likelihood_bins.png")
plt.show()

# -----------------------------------------------------------------------------
# Subplots of mean_log_likelihood vs n_prompt_seqs (per assay)
# -----------------------------------------------------------------------------

fig_ll_vs_np, axes_ll_vs_np = plt.subplots(
    n_rows,
    n_cols,
    figsize=(4 * n_cols, 3 * n_rows),
    sharex=False,  # do not share x-axis as per instructions
    squeeze=False,
)

axes_ll_vs_np_flat = axes_ll_vs_np.flatten()

for i in range(n_rows * n_cols):
    if i >= len(csv_files):
        axes_ll_vs_np_flat[i].axis("off")
        continue

    x_np = all_n_prompt_vals[i]
    y_ll = all_likelihood_vals[i]

    ax_ll = axes_ll_vs_np_flat[i]
    ax_ll.scatter(x_np, y_ll, color="purple", alpha=0.3, s=10)
    # Subtitle = basename of the CSV
    ax_ll.set_title(os.path.basename(csv_files[i]), fontsize=7)

    # Fit spline with ≤2 internal knots
    try:
        sort_idx_local = np.argsort(x_np)
        xs_l, ys_l = x_np[sort_idx_local], y_ll[sort_idx_local]

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
        ax_ll.plot(xs_smooth, ys_smooth, color="blue")
    except Exception as e:
        print(f"Spline (LL vs n_prompt) failed for CSV index {i}: {e}")

    ax_ll.tick_params(axis='both', which='major', labelsize=6)

fig_ll_vs_np.tight_layout()
subplot_path_ll_vs_np = f"{csv_dir}/likelihood_vs_n_prompt_subplots.png"
fig_ll_vs_np.savefig(subplot_path_ll_vs_np)
plt.show()

# -----------------------------------------------------------------------------
# Subplots with RAW spearman, shared likelihood x-axis
# -----------------------------------------------------------------------------

n_rows_raw = n_rows
n_cols_raw = n_cols

fig_raw_shared, axes_raw_shared = plt.subplots(
    n_rows_raw,
    n_cols_raw,
    figsize=(4 * n_cols_raw, 3 * n_rows_raw),
    sharex=True,
    squeeze=False,
)

axes_raw_shared_flat = axes_raw_shared.flatten()

# Re-use global x-limits calculated earlier (global_x_min / global_x_max)
global_spearman_min = -0.05
global_spearman_max = 1.0

for i in range(n_rows_raw * n_cols_raw):
    if i >= len(csv_files):
        axes_raw_shared_flat[i].axis("off")
        continue

    x_rs = all_likelihood_vals[i]
    y_rs = all_spearman_vals[i]

    ax_rs = axes_raw_shared_flat[i]
    ax_rs.scatter(x_rs, y_rs, color="purple", alpha=0.3, s=10)

    # Fit spline (same logic as before)
    try:
        sort_idx_rs = np.argsort(x_rs)
        xs_rs, ys_rs = x_rs[sort_idx_rs], y_rs[sort_idx_rs]

        unique_x_rs = np.unique(xs_rs)
        internal_knots_rs = []
        if len(unique_x_rs) > 12:
            internal_knots_rs = np.quantile(unique_x_rs, [0.1, 0.2, 0.4, 0.8, 0.9]).tolist()
        elif len(unique_x_rs) > 6:
            internal_knots_rs = [np.median(unique_x_rs)]

        if len(internal_knots_rs) > 0:
            spline_rs = LSQUnivariateSpline(xs_rs, ys_rs, t=internal_knots_rs, k=2)
        else:
            spline_rs = UnivariateSpline(xs_rs, ys_rs, k=3, s=0)

        xs_rs_smooth = np.linspace(xs_rs.min(), xs_rs.max(), 200)
        ys_rs_smooth = spline_rs(xs_rs_smooth)
        ax_rs.plot(xs_rs_smooth, ys_rs_smooth, color="blue")

        # Horizontal line at spline maximum
        y_max_rs = ys_rs_smooth.max()
        x_max_rs = xs_rs_smooth[np.argmax(ys_rs_smooth)]
        ax_rs.axhline(y_max_rs, linestyle="--", color="grey", alpha=0.5)
        # Vertical dashed line at exclusion_threshold
        # ax_rs.axvline(exclusion_threshold, linestyle="--", color="grey", alpha=0.5)
        ax_rs.axvline(x_max_rs, linestyle="--", color="grey", alpha=0.5)
    except Exception as e:
        print(f"Spline (raw shared) failed for CSV index {i}: {e}")

    ax_rs.tick_params(axis='both', which='major', labelsize=6)
    ax_rs.set_title(os.path.basename(csv_files[i]), fontsize=7)


for ax_rs in axes_raw_shared_flat[: len(csv_files)]:
    ax_rs.set_ylim(global_spearman_min, global_spearman_max)
    fig_raw_shared.tight_layout()

subplot_path_raw_shared = f"{csv_dir}/spearman_vs_likelihood_raw_subplots_shared_global_y.png"
fig_raw_shared.savefig(subplot_path_raw_shared)
for ax_rs in axes_raw_shared_flat[: len(csv_files)]:
    ax_rs.set_xlim(global_x_min, global_x_max)

subplot_path_raw_shared = f"{csv_dir}/spearman_vs_likelihood_raw_subplots_shared_global_y_and_x.png"
fig_raw_shared.savefig(subplot_path_raw_shared)




fig_raw_shared.tight_layout()

plt.show()

# -----------------------------------------------------------------------------
# NEW: Subplots with RAW spearman, individual likelihood x-axis (non-shared)
# -----------------------------------------------------------------------------

fig_raw, axes_raw = plt.subplots(
    n_rows_raw,
    n_cols_raw,
    figsize=(4 * n_cols_raw, 3 * n_rows_raw),
    sharex=False,
    squeeze=False,
)

axes_raw_flat = axes_raw.flatten()

for i in range(n_rows_raw * n_cols_raw):
    if i >= len(csv_files):
        axes_raw_flat[i].axis("off")
        continue

    x_r = all_likelihood_vals[i]
    y_r = all_spearman_vals[i]

    ax_r = axes_raw_flat[i]
    ax_r.scatter(x_r, y_r, color="purple", alpha=0.3, s=10)

    # Fit spline (same logic as before)
    try:
        sort_idx_r = np.argsort(x_r)
        xs_r, ys_r = x_r[sort_idx_r], y_r[sort_idx_r]

        unique_x_r = np.unique(xs_r)
        internal_knots_r = []
        if len(unique_x_r) > 12:
            internal_knots_r = np.quantile(unique_x_r, [0.1, 0.2, 0.4, 0.8, 0.9]).tolist()
        elif len(unique_x_r) > 6:
            internal_knots_r = [np.median(unique_x_r)]

        if len(internal_knots_r) > 0:
            spline_r = LSQUnivariateSpline(xs_r, ys_r, t=internal_knots_r, k=2)
        else:
            spline_r = UnivariateSpline(xs_r, ys_r, k=3, s=0)

        xs_r_smooth = np.linspace(xs_r.min(), xs_r.max(), 200)
        ys_r_smooth = spline_r(xs_r_smooth)
        ax_r.plot(xs_r_smooth, ys_r_smooth, color="blue")

        y_max_r = ys_r_smooth.max()
        x_max_r = xs_r_smooth[np.argmax(ys_r_smooth)]
        ax_r.axhline(y_max_r, linestyle="--", color="grey", alpha=0.5)
        ax_r.axvline(x_max_r, linestyle="--", color="grey", alpha=0.5)
    except Exception as e:
        print(f"Spline (raw) failed for CSV index {i}: {e}")

    ax_r.tick_params(axis='both', which='major', labelsize=6)
    ax_r.set_title(os.path.basename(csv_files[i]), fontsize=7)

fig_raw.tight_layout()
subplot_path_raw = f"{csv_dir}/spearman_vs_likelihood_raw_subplots.png"
fig_raw.savefig(subplot_path_raw)
plt.show()

# -----------------------------------------------------------------------------
# NEW: Subplots for additional likelihood metrics vs RAW spearman
# -----------------------------------------------------------------------------

metric_configs = [
    (all_likelihood_range_vals, "Likelihood range (max − min)", "likelihood_range_vs_spearman"),
    (all_likelihood_var_vals, "Variance of likelihoods", "likelihood_variance_vs_spearman"),
    (all_likelihood_qcount_vals, "Quantised likelihood count (2 dp)", "likelihood_qcount_vs_spearman"),
]

for metric_list, x_label, fname_stub in metric_configs:
    fig_m, axes_m = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 3 * n_rows),
        sharex=False,
        squeeze=False,
    )
    axes_m_flat = axes_m.flatten()

    for i in range(n_rows * n_cols):
        if i >= len(csv_files):
            axes_m_flat[i].axis("off")
            continue

        x_m = metric_list[i]
        y_m = all_spearman_vals[i]  # RAW spearman values

        ax_m = axes_m_flat[i]
        ax_m.scatter(x_m, y_m, color="purple", alpha=0.3, s=10)

        # Fit smoothing spline (≤2 internal knots)
        try:
            sort_idx_m = np.argsort(x_m)
            xs_m, ys_m = x_m[sort_idx_m], y_m[sort_idx_m]

            unique_x_m = np.unique(xs_m)
            int_knots_m = []
            if len(unique_x_m) > 12:
                int_knots_m = np.quantile(unique_x_m, [0.1, 0.2, 0.4, 0.8, 0.9]).tolist()
            elif len(unique_x_m) > 6:
                int_knots_m = [np.median(unique_x_m)]

            if len(int_knots_m) > 0:
                spline_m = LSQUnivariateSpline(xs_m, ys_m, t=int_knots_m, k=2)
            else:
                spline_m = UnivariateSpline(xs_m, ys_m, k=3, s=0)

            xs_smooth_m = np.linspace(xs_m.min(), xs_m.max(), 200)
            ys_smooth_m = spline_m(xs_smooth_m)
            ax_m.plot(xs_smooth_m, ys_smooth_m, color="blue")
        except Exception as e:
            print(f"Spline ({fname_stub}) failed for CSV index {i}: {e}")

        ax_m.tick_params(axis='both', which='major', labelsize=6)
        ax_m.set_title(os.path.basename(csv_files[i]), fontsize=7)

    fig_m.tight_layout()
    subplot_path_metric = f"{csv_dir}/{fname_stub}_subplots.png"
    fig_m.savefig(subplot_path_metric)
    plt.show()




