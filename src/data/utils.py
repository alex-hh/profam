import numpy as np


def sample_to_max_tokens(
    sequences,
    seed: int = None,
    keep_first: bool = False,
    drop_first: bool = False,
    max_tokens: int = 5000,
):
    rng = np.random.default_rng(seed)
    if keep_first:
        # TODO: might want to allow it to be shuffled
        assert not drop_first
        sampled_sequences = [sequences[0]]
        token_count = len(sampled_sequences[0]) + 2  # bos and eos
    else:
        sampled_sequences = []
        token_count = 2

    shuffled_sequences = sequences[1:] if drop_first or keep_first else sequences
    rng.shuffle(shuffled_sequences)
    for seq in shuffled_sequences:
        if token_count + len(seq) + 1 > max_tokens:
            break
        sampled_sequences.append(seq)
        token_count += len(seq) + 1
    return sampled_sequences
