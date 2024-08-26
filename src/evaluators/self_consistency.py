"""Self-contained self-consistency utils, modified from the FrameDiff implementation, minimising depedencies.

For motif-scaffolding, we probably need to extend the motif scaffolding specific evaluation
in experiments/inference_motif_scaffolding.py, figuring out what inputs are required.

# TODO: possibly use the genie code: https://github.com/aqlaboratory/insilico_design_pipeline (or add PR to switch that code to trasnformers)
"""
import glob
import logging
import os
import subprocess
from typing import Optional

import numpy as np
import pandas as pd
import torch
from proteinsmc.self_consistency import utils
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from transformers.models.esm.openfold_utils.protein import Protein, to_pdb


def convert_outputs_to_pdb(outputs):
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = Protein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs


class SelfConsistencyPDBEvaluator:

    """Lightly modified version of FrameDiff self-consistency evaluation:
    Original code:
    https://github.com/jasonkyuyim/se3_diffusion/blob/master/experiments/inference_se3_diffusion.py

    Removed dependence on a FrameDiff-specific hydra config object and model.
    Also switch esmfold from esm library to transformers for lighter dependencies.

    TODO: package as a library...
    """

    def __init__(
        self,
        device_name: str,  # full name of device, e.g. "cuda:0"
        seq_per_sample: int,
        pmpnn_dir: str,
        sampling_temp: float = 0.1,
        calc_tm_score: bool = True,
        calc_rmsd: bool = True,
        evaluate_native_seq: bool = False,
    ):
        self.device = device_name
        self.seq_per_sample = seq_per_sample
        self.sampling_temp = (
            sampling_temp  # mpnn. TODO check whether 0 for greedy is allowed. NO
        )
        self._pmpnn_dir = pmpnn_dir
        self._log = logging.getLogger(__name__)
        self.calc_tm_score = calc_tm_score
        self.calc_rmsd = calc_rmsd
        self.evaluate_native_seq = evaluate_native_seq

        # TODO check where this gets saved to and cleanup the saved checkpoints from torch if not using
        self._folding_model = EsmForProteinFolding.from_pretrained(
            "facebook/esmfold_v1"
        ).eval()
        self._folding_model.esm = self._folding_model.esm.half()
        self._folding_tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        # self._folding_model = esm.pretrained.esmfold_v1().eval()
        # self._folding_model = self._folding_model.to(self.device)

    def self_consistency_given_mpnn_fasta(
        self,
        mpnn_fasta_path: str,
        reference_pdb_path: str,
        output_dir: str,
        motif_mask: Optional[np.ndarray] = None,
    ):
        # Run ESMFold on each ProteinMPNN sequence and calculate metrics.
        mpnn_results = {
            "sample_path": [],
            "header": [],
            "sequence": [],
        }
        if self.calc_tm_score:
            mpnn_results["tm_score"] = []
        if self.calc_rmsd:
            mpnn_results["rmsd"] = []

        if motif_mask is not None:
            # Only calculate motif RMSD if mask is specified.
            mpnn_results["motif_rmsd"] = []

        esmf_dir = os.path.join(output_dir, "esmf", os.path.basename(mpnn_fasta_path))
        os.makedirs(esmf_dir, exist_ok=True)
        labels, fasta_seqs = utils.read_fasta(mpnn_fasta_path)
        if not self.evaluate_native_seq:
            labels = labels[1:]
            fasta_seqs = fasta_seqs[
                1:
            ]  # skip 'native' seq written as first line by ProteinMPNN
        sample_feats = utils.parse_pdb_feats("sample", reference_pdb_path)
        self._folding_model.to(self.device)
        for i, (header, seq) in enumerate(zip(labels, fasta_seqs)):

            # Run ESMFold
            esmf_sample_path = os.path.join(esmf_dir, f"sample_{i}.pdb")
            _ = self.run_folding(seq, esmf_sample_path)
            esmf_feats = utils.parse_pdb_feats("folded_sample", esmf_sample_path)
            sample_seq = utils.aatype_to_seq(sample_feats["aatype"])

            # Calculate scTM of ESMFold outputs with reference protein
            if self.calc_tm_score:
                _, tm_score = utils.calc_tm_score(
                    sample_feats["bb_positions"],
                    esmf_feats["bb_positions"],
                    sample_seq,
                    sample_seq,
                )
                mpnn_results["tm_score"].append(tm_score)

            if self.calc_rmsd:
                rmsd = utils.calc_aligned_rmsd(
                    sample_feats["bb_positions"], esmf_feats["bb_positions"]
                )
                mpnn_results["rmsd"].append(rmsd)

            if motif_mask is not None:
                sample_motif = sample_feats["bb_positions"][motif_mask]
                of_motif = esmf_feats["bb_positions"][motif_mask]
                motif_rmsd = utils.calc_aligned_rmsd(sample_motif, of_motif)
                mpnn_results["motif_rmsd"].append(motif_rmsd)
            mpnn_results["sample_path"].append(esmf_sample_path)
            mpnn_results["header"].append(header)
            mpnn_results["sequence"].append(seq)

        # Save results to CSV
        mpnn_results = pd.DataFrame(mpnn_results)
        self._folding_model.to("cpu")

        return mpnn_results

    def run_self_consistency_multi(
        self,
        pdb_dir: str,
        output_dir: str,
        motif_mask: Optional[np.ndarray] = None,
    ):
        """We use parse_multiple_chains to repare input.

        self.seq_per_sample sequences per PDB are saved in a PDB-specific fasta.
        """
        pdb_dir = os.path.abspath(pdb_dir)
        pdbs = glob.glob(os.path.join(pdb_dir, "*.pdb"))
        output_path = os.path.join(pdb_dir, "parsed_pdbs.jsonl")

        if not os.path.isdir(os.path.join(output_dir, "seqs")):
            process = subprocess.Popen(
                [
                    "python",
                    f"{self._pmpnn_dir}/helper_scripts/parse_multiple_chains.py",
                    f"--input_path={pdb_dir}",
                    f"--output_path={output_path}",
                ]
            )
            _ = process.wait()
            # Run ProteinMPNN
            num_tries = 0
            ret = -1
            pmpnn_args = [
                "python",
                f"{self._pmpnn_dir}/protein_mpnn_run.py",
                "--out_folder",
                output_dir,
                "--jsonl_path",
                output_path,
                "--num_seq_per_target",
                str(self.seq_per_sample),
                "--sampling_temp",
                f"{self.sampling_temp}",
                "--seed",
                "38",
                "--batch_size",
                "1",
            ]
            # They must be using a modified MPNN implementation that accepts a device argument
            # if "cuda" in self.device:
            #     pmpnn_args.append("--device")
            #     pmpnn_args.append(self.device.split(":")[1])
            while ret < 0:
                # Where do outputs go?
                print("running ProteinMPNN", pmpnn_args)
                try:
                    process = subprocess.Popen(
                        pmpnn_args, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
                    )
                    ret = process.wait()
                except Exception as e:
                    num_tries += 1
                    self._log.info(f"Failed ProteinMPNN. Attempt {num_tries}/5")
                    torch.cuda.empty_cache()
                    if num_tries > 4:
                        raise e

        all_results = []
        csv_path = os.path.join(output_dir, "sc_results.csv")
        if os.path.isfile(csv_path):
            print("Reading existing results")
            return pd.read_csv(csv_path)
        else:
            print("Running ESMFold on ProteinMPNN outputs")
            for pdb_file in pdbs:
                mpnn_fasta_path = os.path.join(
                    output_dir,
                    "seqs",
                    os.path.basename(pdb_file).replace(".pdb", ".fa"),
                )  # sample_0/self_consistency/seqs/sample_1.fa
                pdb_df = self.self_consistency_given_mpnn_fasta(
                    mpnn_fasta_path, pdb_file, output_dir, motif_mask=motif_mask
                )
                pdb_df["pdb"] = pdb_file
                all_results.append(pdb_df)

        df = pd.concat(all_results)
        df.to_csv(csv_path, index=False)
        return df

    def run_self_consistency_single(
        self,
        reference_pdb_path: str,
        output_dir: str,
        motif_mask: Optional[np.ndarray] = None,
    ):
        """Run self-consistency on protein at reference_pdb_path.

        Args:
            decoy_pdb_dir: directory where designed protein files are stored.
            reference_pdb_path: path to reference protein file
            motif_mask: Optional mask of which residues are the motif.

        Returns:
            Writes ProteinMPNN outputs to output_dir/seqs
            Writes ESMFold outputs to output_dir/esmf
            Writes results in output_dir/sc_results.csv
        """
        reference_pdb_path = os.path.abspath(reference_pdb_path)

        # Run ProteinMPNN
        num_tries = 0
        ret = -1
        pmpnn_args = [
            "python",
            f"{self._pmpnn_dir}/protein_mpnn_run.py",
            "--out_folder",
            output_dir,
            "--pdb_path",
            reference_pdb_path,
            # "--jsonl_path",
            # output_path,
            "--num_seq_per_target",
            str(self.seq_per_sample),
            "--sampling_temp",
            "0.1",
            "--seed",
            "38",
            "--batch_size",
            "1",
        ]
        # They must be using a modified MPNN implementation that accepts a device argument
        # if "cuda" in self.device:
        #     pmpnn_args.append("--device")
        #     pmpnn_args.append(self.device.split(":")[1])
        while ret < 0:
            # Where do outputs go?
            print("running ProteinMPNN", pmpnn_args)
            try:
                process = subprocess.Popen(
                    pmpnn_args, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
                )
                ret = process.wait()
            except Exception as e:
                num_tries += 1
                self._log.info(f"Failed ProteinMPNN. Attempt {num_tries}/5")
                torch.cuda.empty_cache()
                if num_tries > 4:
                    raise e

        # print("decopy_pdb_dir", decoy_pdb_dir)   # sample_0/self_consistency/
        # print("reference_pdb_path", reference_pdb_path)  # sample_0/sample_1.pdb
        mpnn_fasta_path = os.path.join(
            output_dir,
            "seqs",
            os.path.basename(reference_pdb_path).replace(".pdb", ".fa"),
        )  # sample_0/self_consistency/seqs/sample_1.fa

        df = self.self_consistency_given_mpnn_fasta(
            mpnn_fasta_path, reference_pdb_path, output_dir, motif_mask=motif_mask
        )
        # TODO: save results to csv?
        return df

    def run_folding(self, sequence, save_path):
        """Run ESMFold on sequence.

        https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/protein_folding.ipynb
        """
        tokenized_input = self._folding_tokenizer(
            [sequence], return_tensors="pt", add_special_tokens=False
        )["input_ids"]

        with torch.no_grad():
            # output = self._folding_model.infer_pdb(sequence)
            output = self._folding_model(tokenized_input.to(self.device))

        pdb = convert_outputs_to_pdb(output)[0]

        with open(save_path, "w") as f:
            f.write(pdb)
        return output
