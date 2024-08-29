import tempfile

import numpy as np
import pytest

from src.data.objects import Protein


@pytest.fixture
def protein_object():
    seq = "".join(np.random.choice(list("ACDEFGHIKLMNPQRSTVWY"), 10))
    coords = np.random.randn(10, 4, 3)
    return Protein(
        sequence=seq,
        accession="",
        backbone_coords=coords,
        plddt=np.random.rand(10) * 100,
    )


def test_pdb_saving_loading(protein_object):
    with tempfile.NamedTemporaryFile(suffix=".pdb") as temp_file:
        protein_object.to_pdb(temp_file.name)
        loaded_protein = Protein.from_pdb(temp_file.name, plddt_from_bfactor=True)
        assert loaded_protein == protein_object
