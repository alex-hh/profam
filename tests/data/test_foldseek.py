import numpy as np
import pandas as pd
import pytest

from src.data.pdb import get_atom_coords_residuewise, load_structure


@pytest.fixture
def foldseek_example():
    """Fixture to load the sample MSA data."""
    df = pd.read_parquet("data/example_data/foldseek_struct/0.parquet")
    return df.iloc[0].to_dict()


def test_foldseek_backbone_loading(foldseek_example):
    ns = foldseek_example["N"]
    cas = foldseek_example["CA"]
    cs = foldseek_example["C"]
    oxys = foldseek_example["O"]
    for ix, (seq, n, ca, c, o, acc) in enumerate(
        zip(
            foldseek_example["sequences"],
            ns,
            cas,
            cs,
            oxys,
            foldseek_example["accessions"],
        )
    ):
        pdbfile = "data/example_data/foldseek_struct/0/AF-{}-F1-model_v4.pdb".format(
            acc, acc
        )
        structure = load_structure(pdbfile, chain="A")
        coords = get_atom_coords_residuewise(["N", "CA", "C", "O"], structure)
        recons_coords = np.zeros_like(coords)
        recons_coords[:, 0] = n.reshape(-1, 3)
        recons_coords[:, 1] = ca.reshape(-1, 3)
        recons_coords[:, 2] = c.reshape(-1, 3)
        recons_coords[:, 3] = o.reshape(-1, 3)
        assert np.allclose(coords, recons_coords)
        assert len(coords) == len(seq)
