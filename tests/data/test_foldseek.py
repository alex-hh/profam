import numpy as np
import pandas as pd
import pytest

from src.data.pdb import get_atom_coords_residuewise, load_structure
from src.data.utils import backbone_coords_from_example


@pytest.fixture
def foldseek_df():
    """Fixture to load the sample MSA data."""
    df = pd.read_parquet("data/example_data/foldseek_struct/0.parquet")
    return df


def test_foldseek_backbone_loading(foldseek_df):
    for _, row in foldseek_df.iterrows():
        foldseek_example = row.to_dict()
        backbone_coords = backbone_coords_from_example(foldseek_example)
        for seq, acc, recons_coords in zip(
            foldseek_example["sequences"],
            foldseek_example["accessions"],
            backbone_coords,
        ):
            pdbfile = "data/example_data/foldseek_struct/0/AF-{}-F1-model_v4.pdb".format(
                acc, acc
            )
            structure = load_structure(pdbfile, chain="A")
            coords = get_atom_coords_residuewise(["N", "CA", "C", "O"], structure)
            assert np.allclose(coords, recons_coords)
            assert len(coords) == len(seq)
