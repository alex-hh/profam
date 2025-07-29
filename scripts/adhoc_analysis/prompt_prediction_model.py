# -----------------------------------------------------------------------------
# Standard library
import os
import glob
import argparse
import pickle

# Third-party
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from scipy.stats import spearmanr
from sklearn.tree import DecisionTreeRegressor

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def load_dms_scores(csv_path, seed=42, max_mutated_sequences=3000):
    dms_df = pd.read_csv(csv_path)
    if max_mutated_sequences is not None and max_mutated_sequences < len(dms_df):
        dms_df = dms_df.sample(n=max_mutated_sequences, random_state=seed)
    return dms_df.DMS_score.values

def _compute_ll_metrics(lls: np.ndarray, precision: int = 2):
    """Compute per-row likelihood statistics.

    Parameters
    ----------
    lls : np.ndarray
        Array of shape (n_rows, n_prompts) with log-likelihoods.
    precision : int, default 2
        Number of decimal places to round to when counting unique values.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - ll_range
        - ll_variance
        - ll_std
        - ll_min
        - ll_max
        - ll_mean
        - ll_median
        - ll_qcount (unique values after rounding)
        - ll_cv (coefficient of variation)
    """
    # Basic statistics
    ll_min = lls.min(axis=1)
    ll_max = lls.max(axis=1)
    ll_mean = lls.mean(axis=1)
    ll_median = np.median(lls, axis=1)
    ll_var = lls.var(axis=1)
    ll_std = np.sqrt(ll_var)

    # Range
    ll_range = ll_max - ll_min

    # Unique rounded count
    quantised_lls = np.round(lls, precision)
    ll_qcount = np.apply_along_axis(lambda x: len(np.unique(x)), 1, quantised_lls)

    # Coefficient of variation (add eps for numerical stability)
    eps = 1e-8
    ll_cv = ll_std / (ll_mean + eps)

    df_metrics = pd.DataFrame({
        "ll_range": ll_range,
        "ll_variance": ll_var,
        "ll_std": ll_std,
        "ll_min": ll_min,
        "ll_max": ll_max,
        "ll_mean": ll_mean,
        "ll_median": ll_median,
        "ll_qcount": ll_qcount,
        "ll_cv": ll_cv,
    })
    return df_metrics


def _load_single_assay(csv_path: str):
    """Load a single ProteinGym assay CSV + companion NPZ and return feature DF.

    The companion NPZ is expected to live next to the CSV and share the same
    basename with suffix ``_lls.npz`` and contain an array named ``lls``.
    """
    npz_path = csv_path.replace(".csv", "_lls.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Expected NPZ next to {csv_path}, got {npz_path}")

    # Load tabular data
    df = pd.read_csv(csv_path)

    # Load likelihoods
    with np.load(npz_path) as data:
        lls = data["lls"]

    if lls.shape[0] != len(df):
        raise ValueError(
            f"Mismatch between CSV rows ({len(df)}) and lls rows ({lls.shape[0]}) in {csv_path}"
        )

    # ------------------------------------------------------------------
    # Compute likelihood-based metrics (per row)
    # ------------------------------------------------------------------
    df_metrics = _compute_ll_metrics(lls)

    # Merge metrics into original df (index aligned)
    df_feat = pd.concat([df.reset_index(drop=True), df_metrics], axis=1)
    df_feat['spearman_centered'] = df_feat['spearman'] - df_feat['spearman'].mean()
    # Keep a reference to the assay file for potential grouping
    df_feat["assay_filename"] = os.path.basename(csv_path)
    assert (abs(df.mean_log_likelihood.values - lls.mean(axis=1)) < 0.0001).all()
    return df_feat


def build_dataset(csv_dir: str):
    """Load all assays in ``csv_dir`` into a single DataFrame of features/target."""
    csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")

    df_all = pd.concat([_load_single_assay(fp) for fp in csv_files], ignore_index=True)

    # ------------------------------------------------------------------
    # Target transformation: centre Spearman WITHIN each assay (mean → 0 per assay)
    # ------------------------------------------------------------------

    # Factorise assay id to a numeric feature (optional)
    df_all["assay_id"] = pd.factorize(df_all["assay_filename"])[0]

    return df_all


def train_xgboost(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Train/test split by entire assays (files) and fit an XGBoost regressor."""
    # Determine unique assays, perform split at assay level
    unique_assays = df["assay_filename"].unique()
    rng = np.random.default_rng(random_state)
    rng.shuffle(unique_assays)

    n_test = max(1, int(len(unique_assays) * test_size))
    test_assays = set(unique_assays[:n_test])

    mask_test = df["assay_filename"].isin(test_assays)
    df_train = df[~mask_test].reset_index(drop=True)
    df_test = df[mask_test].reset_index(drop=True)

    feature_cols = [
        "mean_log_likelihood",
        "n_prompt_seqs",
        # "ll_range",
        # "ll_variance",
        # "ll_std",
        # "ll_min",
        # "ll_max",
    ]

    X_train = df_train[feature_cols]
    y_train = df_train["spearman_centered"]
    X_test = df_test[feature_cols]
    y_test = df_test["spearman_centered"]

    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    # Evaluation
    def _eval(split_name, X, y):
        preds = model.predict(X)
        rmse = mean_squared_error(y, preds, squared=False)
        r2 = r2_score(y, preds)
        print(f"{split_name} — RMSE: {rmse:.4f}, R²: {r2:.4f}")
        return rmse, r2

    print("\nEvaluation metrics (centered Spearman):")
    _eval("Train", X_train, y_train)
    _eval("Test", X_test, y_test)

    return model, feature_cols


def train_decision_tree(
    df: pd.DataFrame,
    max_depth: int,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Train/test split by entire assays (files) and fit a DecisionTree regressor.

    Returns
    -------
    model : DecisionTreeRegressor
        Trained Decision Tree model.
    feature_cols : list[str]
        List of feature columns used for training/prediction.
    """
    # Determine unique assays, perform split at assay level
    unique_assays = df["assay_filename"].unique()
    rng = np.random.default_rng(random_state)
    rng.shuffle(unique_assays)

    n_test = max(1, int(len(unique_assays) * test_size))
    test_assays = set(unique_assays[:n_test])

    mask_test = df["assay_filename"].isin(test_assays)
    df_train = df[~mask_test].reset_index(drop=True)
    df_test = df[mask_test].reset_index(drop=True)

    feature_cols = [
        "mean_log_likelihood",
        "n_prompt_seqs",
        # More features can be added here if desired
    ]

    X_train = df_train[feature_cols]
    y_train = df_train["spearman_centered"]
    X_test = df_test[feature_cols]
    y_test = df_test["spearman_centered"]

    model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)

    # Evaluation
    def _eval(split_name, X, y):
        preds = model.predict(X)
        rmse = mean_squared_error(y, preds, squared=False)
        r2 = r2_score(y, preds)
        print(f"{split_name} — RMSE: {rmse:.4f}, R²: {r2:.4f}")
        return rmse, r2

    print("\nEvaluation metrics (centered Spearman):")
    _eval("Train", X_train, y_train)
    _eval("Test", X_test, y_test)

    return model, feature_cols


def evaluate_topk_predictions(
    model: XGBRegressor,
    df: pd.DataFrame,
    feature_cols: list[str],
    csv_dir: str,
    top_ks: list[int] | None = None,
    dms_root: str = "../data/ProteinGym/DMS_ProteinGym_substitutions",
):
    """Evaluate ensemble likelihood correlation for different *top-k* prompt sets.

    For every assay we rank prompts using the model predictions, build an
    ensemble by averaging log-likelihoods of the *k* best-ranked prompts, and
    compute the Spearman correlation between this ensemble and the ground-truth
    DMS scores.

    Parameters
    ----------
    model : XGBRegressor
        Trained XGBoost model.
    df : pd.DataFrame
        Dataset used for training ‑ must include *assay_filename* and *DMS_id*.
    feature_cols : list[str]
        Columns required to generate model predictions.
    csv_dir : str
        Directory that holds the assay ``.csv`` and companion ``_lls.npz`` files.
    top_ks : list[int], optional
        List of *k* values to evaluate. If *None*, defaults to
        ``[1, 5, 10, 20, 50, 100, 99999]``.
    dms_root : str, optional
        Directory that contains per-assay ground-truth DMS CSVs.

    Returns
    -------
    pd.DataFrame
        One row per (assay, top-k) with the resulting Spearman correlation.
    """
    if top_ks is None:
        top_ks = [1, 5, 10, 15, 20, 35, 50, 100, 99999]

    # Add predictions once to avoid recomputing in each loop
    df_pred = df.copy()
    df_pred["prediction"] = model.predict(df_pred[feature_cols])

    results = []
    for assay_filename, df_assay in df_pred.groupby("assay_filename"):
        csv_path = os.path.join(csv_dir, assay_filename)
        npz_path = csv_path.replace(".csv", "_lls.npz")
        if not os.path.exists(npz_path):
            print(f"[WARN] Missing NPZ for {assay_filename}; skipping.")
            continue

        with np.load(npz_path) as data:
            lls = data["lls"]  # shape: (n_prompts, n_mutated_seqs)

        # Sanity-check alignment
        if lls.shape[0] != len(df_assay):
            print(f"[WARN] Row mismatch in {assay_filename}; skipping.")
            continue
        assert (abs(df_assay.mean_log_likelihood.values - lls.mean(axis=1)) < 0.0001).all()
        # Ranking indices (descending by predicted Spearman)
        sort_idx = np.argsort(-df_assay["prediction"].values)

        # ------------------------------------------------------------------
        # Load ground-truth DMS scores
        # ------------------------------------------------------------------
        if "DMS_id" not in df_assay.columns:
            print(f"[WARN] DMS_id column missing in {assay_filename}; skipping.")
            continue
        dms_id = df_assay["DMS_id"].iloc[0]
        dms_path = os.path.join(dms_root, f"{dms_id}.csv")
        if not os.path.exists(dms_path):
            print(f"[WARN] Missing DMS CSV {dms_path}; skipping {assay_filename}.")
            continue
        dms_scores_path = f"../data/ProteinGym/DMS_ProteinGym_substitutions/{dms_id}.csv"
        dms_scores = load_dms_scores(dms_scores_path, seed=42, max_mutated_sequences=3000)
        try:
            assert abs(df_assay.spearman.iloc[-1] - spearmanr(dms_scores, lls[-1])[0]) < 0.002
        except AssertionError:
            print(f"[WARN] Spearman mismatch in {assay_filename} (DMS={dms_scores.shape[0]}, lls={lls.shape[1]}); skipping.")
            continue
        if dms_scores.shape[0] != lls.shape[1]:
            print(
                f"[WARN] DMS length mismatch in {assay_filename} (DMS={dms_scores.shape[0]}, lls={lls.shape[1]}); skipping."
            )
            continue

        # ------------------------------------------------------------------
        # Evaluate each top-k setting
        # ------------------------------------------------------------------
        for k in top_ks:
            k_eff = min(k, len(sort_idx))  # cannot exceed available prompts
            selected_lls = lls[sort_idx[:k_eff], :]
            ensemble_lls = selected_lls.mean(axis=0)
            rho, _ = spearmanr(ensemble_lls, dms_scores)
            results.append(
                {
                    "assay_filename": assay_filename,
                    "dms_id": dms_id,
                    "k_eff": k_eff,
                    "top_k": k,
                    "spearman_corr": rho,
                }
            )

    return pd.DataFrame(results)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train XGBoost to predict centered Spearman correlation for prompts.")
    parser.add_argument(
        "--csv_dir", 
        type=str, 
        help="Directory containing ProteinGym assay CSVs and NPZs.",
        default="logs/abyoeovl_openfold_fs50_ur90_memmap_251m_copied_2025-06-23_22-18/20250726_173620"
        )
    parser.add_argument("--model_out", type=str, default="xgb_prompt_spearman.pkl", help="Path to write the trained model.")
    parser.add_argument("--test_size", type=float, default=0.3, help="Fraction of assays to reserve for testing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--topk_out", type=str, default="xgb_topk_results.csv", help="CSV to write top-k ensemble evaluation results.")

    args = parser.parse_args()

    print(f"Building dataset from {args.csv_dir} …")
    df = build_dataset(args.csv_dir)

    print(f"Loaded {len(df)} prompt rows from {df['assay_filename'].nunique()} assays.")

    # ------------------------------------------------------------------
    # Decision Tree models (max_depth = 1‒5)
    # ------------------------------------------------------------------
    dt_depths = [1, 2, 3, 4, 5]
    for depth in dt_depths:
        print("\n" + "=" * 70)
        print(f"Training DecisionTreeRegressor (max_depth={depth}) …")
        dt_model, feature_cols = train_decision_tree(
            df,
            max_depth=depth,
            test_size=args.test_size,
            random_state=args.seed,
        )

        # ------------------------------------------------------------------
        # Evaluate ensemble performance with Decision Tree ranking
        # ------------------------------------------------------------------
        print(f"\nEvaluating top-k prompt ensembles with DecisionTreeRegressor (max_depth={depth}) …")
        topk_dt_df = evaluate_topk_predictions(dt_model, df, feature_cols, args.csv_dir)
        if len(topk_dt_df) == 0:
            print("No top-k results were generated – please check warnings above.")
        else:
            grouped_dt = (
                topk_dt_df.groupby("top_k")["spearman_corr"].mean().reset_index().sort_values("top_k")
            )
            print("\nMean Spearman correlation across assays by top-k:")
            for _, row in grouped_dt.iterrows():
                print(f"top-{int(row.top_k):>5}: {row.spearman_corr:.3f}")

    # ------------------------------------------------------------------
    # Train XGBoost model (after Decision Trees)
    # ------------------------------------------------------------------
    model, feature_cols = train_xgboost(df, test_size=args.test_size, random_state=args.seed)

    # ------------------------------------------------------------------
    # Evaluate ensemble performance using model-predicted rankings
    # ------------------------------------------------------------------
    print("\nEvaluating top-k prompt ensembles …")
    topk_df = evaluate_topk_predictions(model, df, feature_cols, args.csv_dir)
    if len(topk_df) == 0:
        print("No top-k results were generated – please check warnings above.")
    else:
        topk_df.to_csv(args.topk_out, index=False)
        grouped = (
            topk_df.groupby("top_k")["spearman_corr"].mean().reset_index().sort_values("top_k")
        )
        print("\nMean Spearman correlation across assays by top-k:")
        for _, row in grouped.iterrows():
            print(f"top-{int(row.top_k):>5}: {row.spearman_corr:.3f}")
        print(f"Top-k results saved to {args.topk_out}")

    # Persist the model + metadata (including per-assay means)
    assay_mean_dict = df.groupby("assay_filename")["spearman"].mean().to_dict()
    with open(args.model_out, "wb") as f:
        pickle.dump({
            "model": model,
            "feature_cols": feature_cols,
            "assay_mean_spearman": assay_mean_dict,
        }, f)

    print(f"Model saved to {args.model_out}")


if __name__ == "__main__":
    main()