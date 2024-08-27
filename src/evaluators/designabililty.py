import os
import shutil
from typing import Dict, List, Optional

import numpy as np

from src.data.objects import Protein, ProteinDocument
from src.evaluators.base import SamplingEvaluator
from src.tools.self_consistency.self_consistency import SelfConsistencyPDBEvaluator


class DesignabilityEvaluator(SamplingEvaluator):
    def __init__(
        self,
        name: str,
        pmpnn_dir: str,
        seed: int = 52,
        num_samples: Optional[int] = None,
        sequences_per_design: int = 1,
        sampling_temp: float = 0.1,
        calc_tm_score: bool = True,
        calc_rmsd: bool = True,
        evaluate_native_seq: bool = False,
        cleanup: bool = True,
    ):
        super().__init__(name, seed=seed, num_samples=num_samples)
        # this runs over saved pdb files containing designs
        self.evaluator = SelfConsistencyPDBEvaluator(
            device_name="cuda:0",
            seq_per_sample=sequences_per_design,
            pmpnn_dir=pmpnn_dir,
            sampling_temp=sampling_temp,
            calc_tm_score=calc_tm_score,
            calc_rmsd=calc_rmsd,
            evaluate_native_seq=evaluate_native_seq,
        )
        self.cleanup = cleanup  # delete all temporary files...

    def _evaluate_samples(
        self,
        protein_document: ProteinDocument,
        samples: List[Protein]
        | np.ndarray,  # for general case we want to accept protein
        output_dir: Optional[str] = None,
        pdb_files: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        if pdb_files is None:
            pdb_files = []
            if isinstance(samples, np.ndarray):
                os.makedirs(os.path.join(output_dir, "generations_pdbs"), exist_ok=True)
                for i in range(samples.shape[0]):
                    protein_coords = samples[i]
                    protein = Protein(
                        sequence="A" * protein_coords.shape[0],
                        backbone_coords=protein_coords,
                    )
                    pdb_file = os.path.join(
                        output_dir, "generations_pdbs", f"sample{i}.pdb"
                    )
                    protein.to_pdb(pdb_file)
                    pdb_files.append(pdb_file)
            else:
                for i, protein in enumerate(samples):
                    pdb_file = os.path.join(
                        output_dir, "generations_pdbs", f"sample{i}.pdb"
                    )
                    protein.to_pdb(pdb_file)
                    pdb_files.append(pdb_file)

        results_df = self.evaluator.run_self_consistency_multi(
            pdb_files,
            output_dir=output_dir,
        )  # pdb, tm_score, rmsd, motif_rmsd, sample_path, seq, header
        metrics = results_df[["tm_score", "rmsd"]].mean().to_dict()
        if self.cleanup:
            shutil.rmtree(output_dir)
        return metrics
