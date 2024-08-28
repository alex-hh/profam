import numpy as np


def backbone_coords_from_example(example):
    ns = example["N"]
    cas = example["CA"]
    cs = example["C"]
    oxys = example["O"]
    coords = []
    for seq, n, ca, c, o in zip(
        example["sequences"],
        ns,
        cas,
        cs,
        oxys,
    ):
        recons_coords = np.zeros((len(seq), 4, 3))
        recons_coords[:, 0] = n.reshape(-1, 3)
        recons_coords[:, 1] = ca.reshape(-1, 3)
        recons_coords[:, 2] = c.reshape(-1, 3)
        recons_coords[:, 3] = o.reshape(-1, 3)
        coords.append(recons_coords)
    return coords
