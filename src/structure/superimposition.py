from Bio.SVDSuperimposer import SVDSuperimposer


def _superimpose_np(reference, coords):
    """
    Superimposes coordinates onto a reference by minimizing RMSD using SVD.

    Args:
        reference:
            [N, 3] reference array
        coords:
            [N, 3] array
    Returns:
        A tuple of [N, 3] superimposed coords and the final RMSD.
    """
    sup = SVDSuperimposer()
    sup.set(reference, coords)
    sup.run()
    return sup.get_transformed(), sup.get_rms()
