import io
from typing import List

import numpy as np
from Bio.PDB import PDBParser
from Bio.SVDSuperimposer import SVDSuperimposer
from tmtools import tm_align
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.residue_constants import atom_order

from src.data.objects import ProteinDocument
from src.evaluators.base import SamplingEvaluator


def calc_tm_score(pos_1, pos_2, seq_1, seq_2):
    # TOOD: check whether it requires only ca or this is a choice
    tm_results = tm_align(pos_1, pos_2, seq_1, seq_2)
    return tm_results.tm_norm_chain1, tm_results.tm_norm_chain2


def _superimpose_np(reference, coords):
    """Superimpose coords onto reference using SVD."""
    sup = SVDSuperimposer()
    sup.set(reference, coords)
    sup.run()
    return sup.get_transformed(), sup.get_rms()


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


class ESMFoldSamplingEvaluator(SamplingEvaluator):
    # TODO: run on single device in multi-gpu setting? or figure out how to distribute?
    # TODO: support caching structure predictions for prompt.
    def __init__(
        self,
        name,
        device,
        max_tokens: int = 8192,
        seed: int = 52,
        prompt_plddt: bool = True,
    ):
        super().__init__(name, seed=seed)
        self.model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").eval()
        self.model.esm = self.model.esm.half()
        self.model = self.model.to("cpu")
        self.device = device
        self.max_tokens = max_tokens  # includes prompt...
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        self.prompt_plddt = prompt_plddt

    def evaluate_samples(self, protein_document: ProteinDocument, samples: List[str]):
        # TODO: add average best TM score or similar to structures in document.
        # https://github.com/blt2114/twisted_diffusion_sampler/blob/968f77111b44e9c711b64e650c41745498ba470d/protein_exp/experiments/inference_se3_diffusion.py#L392
        prompt_plddts = []
        self.model = self.model.to(self.device)
        reference_cas = []
        ref_sequences, _ = self.build_prompt(protein_document)
        ca_index = atom_order["CA"]
        print("Ca index", ca_index)
        for seq in ref_sequences:
            out = self.model.infer(seq)
            print(out.positions.shape)
            # pdb_str = self.model.output_to_pdb(out)[0]
            prompt_plddts.append(np.mean(out.plddt.cpu().numpy()))
            reference_cas.append(out.positions[..., ca_index, :].cpu().numpy())

        sample_plddts = []
        all_tm_scores = []
        for seq in samples:
            out = self.model.infer(seq)
            # pdb_str = self.model.output_to_pdb(out)[0]
            sample_ca = out.positions[..., ca_index, :].cpu().numpy()
            sample_plddts.append(np.mean(out.plddt.cpu().numpy()))
            tm_scores = []
            for ref_seq, ref_ca in zip(ref_sequences, reference_cas):
                # sample_ca = out["structure_module"]["final_atom_positions"].to(self.device)
                # sample_ca, rms = _superimpose_np(ref_ca, sample_ca)
                tm_scores.append(calc_tm_score(ref_ca, sample_ca, ref_seq, seq))
            all_tm_scores.append(tm_scores)

        self.model = self.model.to("cpu")
        return {
            "prompt_plddt": np.mean(prompt_plddts),
            "sample_plddt": np.mean(sample_plddts),
            "min_tm_score": np.mena([min(tm_scores) for tm_scores in all_tm_scores]),
        }
