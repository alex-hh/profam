import io
import os
from typing import List, Optional

import numpy as np
from Bio.PDB import PDBParser
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils import atom14_to_atom37
from transformers.models.esm.openfold_utils.residue_constants import (
    atom_order,
    restypes_with_x,
)

from src.data.objects import Protein, ProteinDocument
from src.evaluators.base import SamplingEvaluator
from src.sequence.utils import decode_tokens
from src.structure.superimposition import rmsd, tm_score


def structure_from_pdb_str(pdb_str: str):
    with io.StringIO(pdb_str) as pdb_fh:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(id="none", file=pdb_fh)
    return structure


def extract_plddts(structure):
    ca_b_factors = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.name == "CA":
                        ca_b_factors.append(atom.bfactor)
    return ca_b_factors


def load_residues(pdb_file):
    parser = PDBParser()
    structure = parser.get_structure(id=None, file=pdb_file)
    residues = [res for res in structure[0]["A"]]
    return residues


def esmfold_output_to_proteins(output):
    # c.f. ESMForProteinFolding.output_to_pdb
    output = {k: v.to("cpu").numpy() for k, v in output.items()}
    proteins = []
    final_atom_positions = atom14_to_atom37(output["positions"][-1], output)
    final_atom_mask = output["atom37_atom_exists"]
    for i in range(output["aatype"].shape[0]):
        aa = output["aatype"][i]
        sequence = decode_tokens(aa, restypes_with_x)
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        backbone_atom_ids = [atom_order[atom] for atom in ["N", "CA", "C", "O"]]
        pred = Protein(
            sequence=sequence,
            backbone_coords=pred_pos[:, backbone_atom_ids],
            backbone_coords_mask=mask[:, backbone_atom_ids],
            plddt=output["plddt"][i].mean(-1) * 100.0,  # average over atoms per residue
        )
        proteins.append(pred)
    return proteins


class ESMFoldInverseFoldingEvaluator(SamplingEvaluator):
    # TODO: run on single device in multi-gpu setting? or figure out how to distribute?
    # TODO: support caching structure predictions for prompt.
    def __init__(
        self,
        name,
        num_samples: Optional[int] = None,
        prompt_plddt: bool = True,
        half_precision: bool = False,
        use_precomputed_reference_structures: bool = True,
        save_structures: bool = False,  # TODO: check whether saved structures correspond to samples and re-use if so.
        # force_recompute_structures: bool = False, in this case can manually delete saved structures.
        verbose: bool = False,
        max_length: int = 512,  # TODO look into cpu offloading...
        **kwargs,
    ):
        super().__init__(name, num_samples=num_samples, **kwargs)
        # TODO: defer loading model until first call to evaluate_samples
        self.esmfold = None
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        self.prompt_plddt = prompt_plddt
        self.half_precision = half_precision
        self.use_precomputed_reference_structures = use_precomputed_reference_structures
        self.save_structures = save_structures
        self.max_length = max_length  # TODO: we can actually enforce this on sampling.
        self.verbose = verbose

    def _load_model(self, device):
        self.esmfold = EsmForProteinFolding.from_pretrained(
            "facebook/esmfold_v1"
        ).eval()
        self.esmfold = self.esmfold.to(device)
        if self.half_precision:
            print("Using half precision")
            self.esmfold = self.esmfold.half()

    def _load_precomputed_reference_structure(self, output_dir, representative):
        target_file = os.path.join(output_dir, "target.pdb")
        if os.path.exists(target_file):
            ref_prot = Protein.from_pdb(target_file, bfactor_is_plddt=True)
            if representative.sequence == ref_prot.sequence:
                return ref_prot, True
            return ref_prot, False
        else:
            return None, False

    def _load_precomputed_sample_structures(self, output_dir, samples):
        sample_prots = []
        is_valid = True
        for i, seq in enumerate(samples):
            sample_file = os.path.join(output_dir, f"sample_{i}.pdb")
            if os.path.exists(sample_file):
                prot = Protein.from_pdb(sample_file, bfactor_is_plddt=True)
                sample_prots.append(prot)
                if prot.sequence != seq:
                    is_valid = False
            else:
                is_valid = False
                break
        return sample_prots, is_valid

    def _evaluate_samples(
        self,
        prompt: ProteinDocument,
        protein_document: ProteinDocument,
        samples: List[str],
        output_dir: Optional[str] = None,
        device: Optional[str] = None,
    ):
        assert device is not None
        if self.esmfold is None:
            self._load_model(device)
        self.esmfold.to(device)

        # TODO: really we should compare to ProteinDocument and not to prompt, which may differ...
        # TODO: add average best TM score or similar to structures in document.
        # https://github.com/blt2114/twisted_diffusion_sampler/blob/968f77111b44e9c711b64e650c41745498ba470d/protein_exp/experiments/inference_se3_diffusion.py#L392
        if self.save_structures:
            os.makedirs(output_dir, exist_ok=True)

        assert len(prompt) > 0

        representative = protein_document.representative
        # infer prompt structures
        if (
            not self.use_precomputed_reference_structures
            or representative.backbone_coords is None
        ):
            ref_prot, is_valid = self._load_precomputed_reference_structure(
                output_dir, representative
            )
            if not is_valid and len(representative.sequence) <= self.max_length:
                assert not representative.sequence.endswith("|")
                out = self.esmfold.infer(seq)
                ref_prot = esmfold_output_to_proteins(out)[0]
                if self.save_structures:
                    ref_prot.to_pdb_file(os.path.join(output_dir, "target.pdb"))
            else:
                ref_prot = None
        else:
            ref_prot = representative
            # handle interleaving
            if ref_prot.sequence[-1] == "|":
                ref_prot = ref_prot.slice_arrays(0, len(ref_prot.sequence) - 1)

        num_samples_greater_than_max_length = 0
        (
            sample_prots,
            precomputed_samples_are_valid,
        ) = self._load_precomputed_sample_structures(output_dir, samples)
        if not precomputed_samples_are_valid:
            sample_prots = []
            for i, seq in enumerate(samples):
                if len(seq) <= self.max_length:
                    out = self.esmfold.infer(seq)
                    prot = esmfold_output_to_proteins(out)[0]
                    sample_prots.append(prot)
                    if self.save_structures:
                        prot.to_pdb_file(os.path.join(output_dir, f"sample_{i}.pdb"))
                else:
                    num_samples_greater_than_max_length += (
                        1  # n.b. this will be wrong if we re-use
                    )

        tm_scores = []
        rmsds = []
        for prot in sample_prots:
            if ref_prot is not None:
                tm_scores.append(tm_score(prot, ref_prot))
                rmsds.append(rmsd(ref_prot, prot))

        self.esmfold = self.esmfold.to("cpu")
        if self.verbose:
            print(
                f"Sample PLDDT: {np.mean([np.mean(prot.plddt) for prot in sample_prots])} "
                f"TM Score: {np.mean(tm_scores)} RMSD: {np.mean(rmsds)}",
                flush=True,
            )
        metrics = {
            "sample_plddt": np.mean([np.mean(prot.plddt) for prot in sample_prots]),
            "sample_lens": np.mean([len(prot) for prot in sample_prots]),
        }
        if not precomputed_samples_are_valid:
            # only computed correctly if running for the first time. #TODO: fix
            metrics[
                "num_samples_greater_than_max_length"
            ] = num_samples_greater_than_max_length
        if ref_prot is not None:
            metrics["prompt_lens"] = len(ref_prot)
            metrics["mean_tm_score"] = np.mean(tm_scores)
            metrics["mean_rmsd"] = np.mean(rmsds)
            if ref_prot.plddt is not None:
                metrics["prompt_plddt"] = np.mean(ref_prot.plddt)
        return metrics


class ESMFoldSamplingEvaluator(SamplingEvaluator):
    # TODO: run on single device in multi-gpu setting? or figure out how to distribute?
    # TODO: support caching structure predictions for prompt.
    def __init__(
        self,
        name,
        num_samples: Optional[int] = None,
        prompt_plddt: bool = True,
        half_precision: bool = False,
        use_precomputed_reference_structures: bool = True,
        save_structures: bool = False,
        verbose: bool = False,
        max_length: int = 512,  # TODO look into cpu offloading...
        **kwargs,
    ):
        super().__init__(name, num_samples=num_samples, **kwargs)
        # TODO: defer loading model until first call to evaluate_samples
        self.esmfold = None
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        self.prompt_plddt = prompt_plddt
        self.half_precision = half_precision
        self.use_precomputed_reference_structures = use_precomputed_reference_structures
        self.save_structures = save_structures
        self.max_length = max_length  # TODO: we can actually enforce this on sampling.
        self.verbose = verbose

    def _load_model(self, device):
        self.esmfold = EsmForProteinFolding.from_pretrained(
            "facebook/esmfold_v1"
        ).eval()
        self.esmfold = self.esmfold.to(device)
        if self.half_precision:
            print("Using half precision")
            self.esmfold = self.esmfold.half()

    def _evaluate_samples(
        self,
        prompt: ProteinDocument,
        protein_document: ProteinDocument,
        samples: List[str],
        output_dir: Optional[str] = None,
        device: Optional[str] = None,
    ):
        prompt_accessions = prompt.accessions
        # we can't compare to prompt directly as it has had transforms applied - noise, rotation, rescaling...
        raw_prompt_document = protein_document.filter(
            lambda prot: prot.accession in prompt_accessions
        )
        assert device is not None
        if self.esmfold is None:
            self._load_model(device)
        self.esmfold.to(device)

        # TODO: really we should compare to ProteinDocument and not to prompt, which may differ...
        # TODO: add average best TM score or similar to structures in document.
        # https://github.com/blt2114/twisted_diffusion_sampler/blob/968f77111b44e9c711b64e650c41745498ba470d/protein_exp/experiments/inference_se3_diffusion.py#L392
        if self.save_structures:
            os.makedirs(output_dir, exist_ok=True)

        assert len(prompt) > 0

        prompt_prots = []
        for i, seq in enumerate(raw_prompt_document.sequences):
            if len(seq) == 0:  # unconditional sampling
                continue
            # infer prompt structures
            if (
                not self.use_precomputed_reference_structures
                or raw_prompt_document.backbone_coords is None
            ):
                if len(seq) <= self.max_length:
                    # handle interleaving
                    if seq[-1] == "|":
                        seq = seq[:-1]
                    out = self.esmfold.infer(seq)
                    prot = esmfold_output_to_proteins(out)[0]
                    if self.save_structures:
                        prot.to_pdb_file(os.path.join(output_dir, f"prompt_{i}.pdb"))
                    prompt_prots.append(prot)
            else:
                prot = raw_prompt_document[i]
                # handle interleaving
                if prot.sequence[-1] == "|":
                    prot = prot.slice_arrays(0, len(prot.sequence) - 1)
                prompt_prots.append(prot)

        sample_prots = []
        all_tm_scores = []
        num_samples_greater_than_max_length = 0
        for i, seq in enumerate(samples):
            if len(seq) <= self.max_length:
                out = self.esmfold.infer(seq)
                prot = esmfold_output_to_proteins(out)[0]
                sample_prots.append(prot)
                if len(prompt_prots) > 0:
                    tm_scores = []
                    for ref_prot in prompt_prots:
                        tm_scores.append(tm_score(prot, ref_prot))
                    all_tm_scores.append(tm_scores)
            else:
                num_samples_greater_than_max_length += 1
            if self.save_structures:
                prot.to_pdb_file(os.path.join(output_dir, f"sample_{i}.pdb"))

        self.esmfold = self.esmfold.to("cpu")
        if self.verbose:
            print(
                f"Sample PLDDT: {np.mean([np.mean(prot.plddt) for prot in sample_prots])} "
                f"TM Score: {np.mean([np.mean(tm_scores) for tm_scores in all_tm_scores])}",
                flush=True,
            )
        metrics = {
            "sample_plddt": np.mean([np.mean(prot.plddt) for prot in sample_prots]),
            "sample_lens": np.mean([len(prot) for prot in sample_prots]),
            "num_samples_greater_than_max_length": num_samples_greater_than_max_length,
        }
        if len(prompt_prots) > 0:
            metrics.update(
                {
                    "prompt_plddt": np.mean(
                        [np.mean(prot.plddt) for prot in prompt_prots]
                    ),
                    "prompt_lens": np.mean([len(prot) for prot in prompt_prots]),
                    "min_tm_score": np.mean(
                        [min(tm_scores) for tm_scores in all_tm_scores]
                    ),
                    "max_tm_score": np.mean(
                        [max(tm_scores) for tm_scores in all_tm_scores]
                    ),
                    "mean_tm_score": np.mean(
                        [np.mean(tm_scores) for tm_scores in all_tm_scores]
                    ),
                }
            )
        return metrics
