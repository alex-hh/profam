import numpy as np

from src.constants import aa_letters


def hamming_distance(seq_a, seq_b, ignore_gaps=False, include_flanking=False):
    if ignore_gaps:
        d = sum([a != b for (a, b) in zip(seq_a, seq_b) if a != "-" and b != "-"])
    else:
        d = sum([a != b for (a, b) in zip(seq_a, seq_b)])
    if include_flanking:
        if ignore_gaps:
            raise NotImplementedError()
        d += abs(len(seq_a) - len(seq_b))
    return d


def sequence_identity(seq_a, seq_b, ignore_gaps=False):
    dist = hamming_distance(seq_a, seq_b, ignore_gaps=ignore_gaps)
    assert len(seq_a) == len(seq_b)
    if ignore_gaps:
        raise NotImplementedError()
        # non_gap_A = len([aa for aa in seq_a if aa != "-"])
        # non_gap_B = len([aa for aa in seq_b if aa != "-"])
        # seq_id = dist / min(non_gap_A, non_gap_B)
    else:
        if dist == len(seq_a):
            seq_id = 0
        else:
            seq_id = (len(seq_a) - dist) / len(seq_a)
    return seq_id


def blosum_distance(seq_a, seq_b):
    from biotite.sequence import align

    blosum = align.SubstitutionMatrix.dict_from_db("BLOSUM62")
    score = 0  # higher is closer
    for (aa_a, aa_b) in zip(seq_a, seq_b):
        if aa_a != aa_b and aa_a != "-" and aa_b != "-":
            score += blosum[(aa_a, aa_b)]
    return score


def random_seq(L):
    return "".join(np.random.choice(aa_letters, L, replace=True))


def decode_tokens(numeric, aa_toks):
    # c.f. one of my esm utils?
    id2aa = {i: aa for i, aa in enumerate(aa_toks)}
    tok_arr = np.vectorize(id2aa.__getitem__)(numeric)
    if numeric.ndim == 1:
        return "".join(tok_arr)
    return ["".join(toks) for toks in tok_arr]
