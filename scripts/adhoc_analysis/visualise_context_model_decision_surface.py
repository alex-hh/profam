import argparse
import os
import pickle
import sys
import importlib.util

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _import_prompt_prediction_model(module_path: str):
    """Dynamically import *prompt_prediction_model.py* that lives next to this file.

    We cannot rely on the directory being a Python package, so we load the
    module via its file path.
    """
    spec = importlib.util.spec_from_file_location("prompt_prediction_model", module_path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec.loader is not None  # mypy
    spec.loader.exec_module(module)  # type: ignore[misc]
    return module


# -----------------------------------------------------------------------------
# Main visualisation routine
# -----------------------------------------------------------------------------

def plot_decision_surface(
    model_path: str,
    csv_dir: str,
    grid_size: int = 200,
    output_path: str | None = None,
    show_samples: bool = False,
):
    """Generate and plot the 2-D decision surface of a trained 2-feature XGB model.

    Parameters
    ----------
    model_path : str
        Path to the pickle produced by *prompt_prediction_model.py* (must hold
        a dict with keys ``model`` and ``feature_cols``).
    csv_dir : str
        Directory that contains the ProteinGym assay CSV + NPZ files.
        This is only used to rebuild the dataset so we can determine the min/max
        range for each feature.
    grid_size : int, default 200
        Number of points along **each** feature axis of the grid. The total
        evaluations will therefore be *grid_size²*.
    output_path : str, optional
        If given, the resulting figure is saved to this path in addition to
        being shown interactively.
    show_samples : bool, default False
        Whether to overlay the original sample locations (as semi-transparent
        dots) on top of the heatmap.
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Cannot find model pickle at '{model_path}'.")

    # ------------------------------------------------------------------
    # Load trained model and metadata
    # ------------------------------------------------------------------
    with open(model_path, "rb") as f:
        payload = pickle.load(f)

    if not isinstance(payload, dict):
        raise ValueError("Expected the pickle to contain a dict with keys 'model' and 'feature_cols'.")

    model = payload.get("model")
    feature_cols: list[str] = payload.get("feature_cols")  # type: ignore[assignment]

    if model is None or feature_cols is None:
        raise KeyError("The pickle does not contain both 'model' and 'feature_cols'.")
    if len(feature_cols) != 2:
        raise ValueError(
            f"This visualisation expects exactly 2 feature columns, got {len(feature_cols)}: {feature_cols}"
        )

    # ------------------------------------------------------------------
    # Import *prompt_prediction_model.py* dynamically and build dataset
    # ------------------------------------------------------------------
    this_dir = os.path.dirname(os.path.abspath(__file__))
    ppm_path = os.path.join(this_dir, "prompt_prediction_model.py")
    ppm = _import_prompt_prediction_model(ppm_path)

    df = ppm.build_dataset(csv_dir)  # type: ignore[attr-defined]

    # Feature ranges (use min/max across *all* assays)
    f0_min, f0_max = df[feature_cols[0]].min(), df[feature_cols[0]].max()
    f1_min, f1_max = df[feature_cols[1]].min(), df[feature_cols[1]].max()

    # Generate grid – use linspace to cover the closed interval inclusively
    x = np.linspace(f0_min, f0_max, grid_size)
    y = np.linspace(f1_min, f1_max, grid_size)
    xx, yy = np.meshgrid(x, y)

    grid_df = pd.DataFrame({
        feature_cols[0]: xx.ravel(),
        feature_cols[1]: yy.ravel(),
    })

    # Model expects exactly these two columns in the same order
    zz = model.predict(grid_df[feature_cols]).reshape(xx.shape)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    plt.figure(figsize=(8, 6))

    # imshow provides a convenient way to plot the 2-D array; we set extent so
    # that the axes are in feature space units rather than pixel indices.
    im = plt.imshow(
        zz,
        origin="lower",
        extent=(f0_min, f0_max, f1_min, f1_max),
        aspect="auto",
        cmap="viridis",
    )
    cbar = plt.colorbar(im)
    cbar.set_label("Predicted centered Spearman")

    plt.xlabel(feature_cols[0])
    plt.ylabel(feature_cols[1])
    plt.title("XGB decision surface (2-feature model)")

    if show_samples:
        plt.scatter(
            df[feature_cols[0]],
            df[feature_cols[1]],
            c="white",
            alpha=0.3,
            edgecolor="none",
            s=10,
        )

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300)
        print(f"Decision surface saved to '{output_path}'.")

    plt.show()


# -----------------------------------------------------------------------------
# CLI entry-point
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualise the 2-feature XGB model's decision surface.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="xgb_prompt_spearman.pkl",
        help="Path to the pickled model created by *prompt_prediction_model.py*.",
    )

    # Optionally train a Decision Tree model on-the-fly before visualising
    parser.add_argument(
        "--train_dt",
        action="store_true",
        help="Train a depth-5 DecisionTreeRegressor (as implemented in prompt_prediction_model.py) and visualise its decision surface.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of assays to reserve for testing when training the Decision Tree (only used with --train_dt).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset splitting / training (only used with --train_dt).",
    )
    parser.add_argument(
        "--csv_dir",
        type=str,
        default="logs/abyoeovl_openfold_fs50_ur90_memmap_251m_copied_2025-06-23_22-18/20250726_173620",
        help="Directory that contains the ProteinGym assay CSV + NPZ files (same as used for training).",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=200,
        help="Number of points along each axis of the grid (default: 200 ⇒ 40k evaluations).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="two_feature_xgb_decision_surface.png",
        help="File path to write the resulting PNG; if omitted, the image is only shown interactively.",
    )
    parser.add_argument(
        "--show_samples",
        action="store_true",
        help="Overlay the original sample locations on top of the heatmap (for reference).",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Optional: train a Decision Tree model before visualisation
    # ------------------------------------------------------------------
    if True: # args.train_dt:
        print("\nTraining DecisionTreeRegressor (max_depth=5) …")

        # Dynamically import helper module
        this_dir = os.path.dirname(os.path.abspath(__file__))
        ppm_path = os.path.join(this_dir, "prompt_prediction_model.py")
        ppm = _import_prompt_prediction_model(ppm_path)

        # Build dataset and train model
        df_full = ppm.build_dataset(args.csv_dir)  # type: ignore[attr-defined]
        dt_model, feature_cols = ppm.train_decision_tree(  # type: ignore[attr-defined]
            df_full,
            max_depth=5,
            test_size=args.test_size,
            random_state=args.seed,
        )

        # Persist the freshly trained model so that plot_decision_surface can load it
        with open(args.model_path, "wb") as f:
            pickle.dump({"model": dt_model, "feature_cols": feature_cols}, f)

        print(f"Decision Tree model saved to '{args.model_path}'.")

    plot_decision_surface(
        model_path=args.model_path,
        csv_dir=args.csv_dir,
        grid_size=args.grid_size,
        output_path=args.output,
        show_samples=args.show_samples,
    )


if __name__ == "__main__":
    main()
