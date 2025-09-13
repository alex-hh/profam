"""
Created by Jude Wells 2025-09-12

This script evaluates generated sequences from a given model.
We assume that you have already generated the sequences and they are in fasta format.
We assume that you have the target structures (representative from each cluster)
"""

import glob
import os
from Bio import SeqIO
from src.utils.evaluation_utils import sequence_only_evaluation
import pandas as pd
import numpy as np
from src.data.objects import Protein
from src.structure.superimposition import tm_score, lddt
from src.utils.evaluation_utils import pairwise_sequence_identity
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt


def make_structure_sequence_similarity_plots(csv_path):
    df_mode = pd.read_csv(csv_path)
    structure_metrics = ['tm_max', 'lddt_max']
    for structure_metric in structure_metrics:
        x = df_mode["seq_identity_max"].to_numpy(dtype=float)
        y = df_mode[structure_metric].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x_valid = x[mask]
        y_valid = y[mask]

        if x_valid.size == 0:
            continue

        plt.scatter(x_valid, y_valid, s=12, alpha=0.5, label="samples")

        try:
            smoothed = lowess(y_valid, x_valid, frac=0.3, return_sorted=True)
            plt.plot(smoothed[:, 0], smoothed[:, 1], color="crimson", linewidth=2, label="LOWESS")
        except Exception:
            pass

        plt.xlabel("Max sequence identity prompt")
        plt.ylabel(structure_metric)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(f"{csv_path.replace('.csv', f'_{structure_metric}.png')}")
        plt.close()
        



if __name__ == "__main__":

    make_structure_sequence_similarity_plots("/mnt/disk2/cath_plm/sampling_results/colabfold_outputs/poet_structural_evaluation.csv")