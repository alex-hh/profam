import io
from typing import List

import numpy as np
from Bio.PDB import PDBParser
from Bio.SVDSuperimposer import SVDSuperimposer
from transformers import AutoTokenizer, EsmForProteinFolding

from src.data.objects import ProteinDocument
from src.evaluators.base import SamplingEvaluator


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
    def __init__(self, device, max_tokens: int):
        self.model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").eval()
        self.model.esm = self.model.esm.half()
        self.model = self.model.to("cpu")
        self.device = device
        self.max_tokens = max_tokens  # includes prompt...
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")

    def build_prompt(self, protein_document: ProteinDocument):
        # TODO: we're going to need to subsample to max tokens
        pass

    def evaluate_samples(self, protein_document: ProteinDocument, samples: List[str]):
        # TODO: add average best TM score or similar to structures in document.
        prompt_plddts = []
        self.model = self.model.to(self.device)
        assert protein_document.prompt_indices is not None  # set during build prompt
        sequence_prompt = [
            protein_document.sequences[i] for i in protein_document.prompt_indices
        ]
        for seq in sequence_prompt:
            out = self.model.infer(seq)
            # pdb_str = self.model.output_to_pdb(out)[0]
            prompt_plddts.append(np.mean(out.plddt.cpu().numpy()))

        sample_plddts = []
        for seq in samples:
            out = self.model.infer(seq)
            # pdb_str = self.model.output_to_pdb(out)[0]
            sample_plddts.append(np.mean(out.plddt.cpu().numpy()))

        self.model = self.model.to("cpu")
        return {
            "prompt_plddt": np.mean(prompt_plddts),
            "sample_plddt": np.mean(sample_plddts),
        }
