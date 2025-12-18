from typing import Callable, Optional

import numpy as np

from src.data.objects import Protein, ProteinDocument
from src.data.tokenizers import ProFamTokenizer


def convert_aligned_sequence_adding_positions(
    seq,
    keep_gaps=True,
    keep_insertions=True,
    to_upper=False,
    use_msa_pos: bool = True,
):
    """
    # N.B. defaults currently raise an exception.

    Get positions relative to sequence.
    For alignments, if use_msa_pos is True, the positions are relative to the alignment columns
    (match states). Insertions have the same position index as the previous match state.

    If use_msa_pos is False, or the sequence is unaligned,
    positions are relative to the retained sequence - ignored insertions dont contribute

    For both raw and aligned sequences, the first non-insertions should have position 1.

    N.B. currently there is ambiguity between position encoding for a gap then insert
    and a match state. we require a binary mask to resolve.
    """
    match_index = 0  # 0 for inserts before first match state
    positions = []
    is_match = []
    sequence = ""

    # if not use_msa_pos:
    #     return seq, list(range(1, len(seq+1))), [True] * len(seq)

    if keep_insertions:
        assert to_upper, "If keeping insertions should convert to upper case"
    for aa in seq:
        if keep_gaps or aa != "-":
            if aa == ".":
                # dont keep gaps in insert columns: we can modify later if we ever want to use
                continue
            # at this point we have any amino acid character (match or insert) or a match gap
            # TODO: check for valid characters
            upper = aa.upper()
            if upper == aa or keep_insertions:
                # increment first so that insert corresponds to prev match state
                if upper == aa and aa != ".":  # includes case where aa is "-"
                    match_index += 1
                    is_match.append(True)
                else:
                    assert aa != "."
                    # insertion
                    if not use_msa_pos:
                        match_index += 1
                    is_match.append(False)
                positions.append(match_index)
                sequence += upper
            # otherwise we're not keeping insertions in which case we pass

        elif aa == "-":
            if use_msa_pos:
                match_index += 1  # keep_gaps is False so we dont add to sequence but still increment match_index

    assert len(positions) == len(
        sequence
    ), f"positions length {len(positions)} != sequence length {len(sequence)}"
    assert len(sequence) == len(
        is_match
    ), f"sequence length {len(sequence)} != is_match length {len(is_match)}"
    return sequence, positions, is_match


def convert_raw_sequence_adding_positions(seq):
    return seq, list(range(1, len(seq) + 1)), [True] * len(seq)


def _get_truncated_slice(seq_length, max_length, rnd):
    if seq_length > max_length:
        truncation_start = rnd.randint(0, seq_length - max_length)
        truncation_end = truncation_start + max_length
        return slice(truncation_start, truncation_end)
    else:
        return slice(None)


def preprocess_raw_sequences_sampling_to_max_tokens(
    proteins: ProteinDocument,
    tokenizer: ProFamTokenizer,
    max_tokens: Optional[int] = None,
    shuffle: bool = True,
    rng: Optional[np.random.Generator] = None,
    drop_first: bool = False,
    keep_first: bool = False,
    **kwargs,
) -> ProteinDocument:
    """
    Sample proteins to fit within a maximum token limit while adding positions and standardising sequences.

    Sequence converter may need to differ depening on whether raw sequences are in a2m/a3m format or standard fasta format.

    Args:
        proteins: ProteinDocument containing the proteins to sample from.
        max_tokens: Maximum number of tokens allowed.
        tokenizer: Optional ProFamTokenizer for accurate token counting.
        shuffle: Whether to shuffle the proteins before sampling.
        seed: Random seed for shuffling.
        drop_first: Whether to drop the first protein before sampling.
        keep_first: Whether to always keep the first protein in the sample.
        extra_tokens_per_document: Number of extra tokens per document.

    Returns:
        A new ProteinDocument containing the sampled proteins.
    """
    extra_tokens_per_protein = 1  # separator token
    extra_tokens_per_document = tokenizer.num_start_tokens

    rnd = np.random if rng is None else rng
    if drop_first:
        proteins = proteins[1:]

    if shuffle:
        perm = rnd.permutation(len(proteins))
        if keep_first:
            perm = np.concatenate(([0], perm[perm != 0]))
    else:
        perm = np.arange(len(proteins))

    # todo: could store precomputed sequence lengths on object...but would need to keep updated.
    new_sequence_lengths = np.array(
        [len(seq) + extra_tokens_per_protein for seq in proteins.sequences]
    )[perm]
    max_length = np.max(new_sequence_lengths)
    truncated_sequence_lengths = np.minimum(new_sequence_lengths, max_length)
    cumsum_lengths = extra_tokens_per_document + np.cumsum(truncated_sequence_lengths)
    if max_tokens is not None:
        endpoint = np.searchsorted(
            cumsum_lengths, max_tokens
        )  # position at which max_tokens is inserted to sort array - so we can actually include next element and truncate
        if endpoint > 0 and endpoint < len(proteins):
            final_element_tokens = (
                max_tokens - cumsum_lengths[endpoint - 1] - extra_tokens_per_protein
            )  # cumsum lengths include extra tokens
            if final_element_tokens > extra_tokens_per_protein:
                effective_endpoint = endpoint + 1  # add a truncated element
            else:
                effective_endpoint = endpoint
        elif endpoint >= len(proteins):
            effective_endpoint = len(proteins)
            final_element_tokens = new_sequence_lengths[-1]
        else:
            # endpoint == 0
            final_element_tokens = (
                max_tokens - extra_tokens_per_document - extra_tokens_per_protein
            )
            effective_endpoint = 1  # add a truncated element
        new_proteins = proteins[perm[:effective_endpoint]]
        assert final_element_tokens >= 0

        array_slices = [slice(None)] * effective_endpoint

        if effective_endpoint <= len(proteins) and final_element_tokens > 0:
            # TODO: rng seed this
            assert len(array_slices) == effective_endpoint
            final_array_slice = _get_truncated_slice(
                new_sequence_lengths[effective_endpoint - 1], final_element_tokens, rnd
            )
            array_slices[-1] = final_array_slice

        assert len(array_slices) == len(new_proteins)

    else:
        new_proteins = proteins[perm]
        array_slices = [slice(None)] * len(new_proteins)

    new_proteins = new_proteins.slice_arrays(array_slices)
    return new_proteins


def preprocess_aligned_sequences_sampling_to_max_tokens(
    proteins: ProteinDocument,
    tokenizer: ProFamTokenizer,
    sequence_converter: Callable,
    max_tokens: Optional[int] = None,
    shuffle: bool = True,
    rng: Optional[np.random.Generator] = None,
    drop_first: bool = False,
    keep_first: bool = False,
    allow_partial_sequence: bool = False,
    **kwargs,
) -> ProteinDocument:
    """
    Sample proteins to fit within a maximum token limit while adding positions and standardising sequences.

    Sequence converter may need to differ depening on whether raw sequences are in a2m/a3m format or standard fasta format.

    Args:
        proteins: ProteinDocument containing the proteins to sample from.
        max_tokens: Maximum number of tokens allowed.
        tokenizer: Optional ProFamTokenizer for accurate token counting.
        shuffle: Whether to shuffle the proteins before sampling.
        seed: Random seed for shuffling.
        drop_first: Whether to drop the first protein before sampling.
        keep_first: Whether to always keep the first protein in the sample.
        extra_tokens_per_document: Number of extra tokens per document.

    Returns:
        A new ProteinDocument containing the sampled proteins.
    """
    extra_tokens_per_protein = 1  # separator token
    extra_tokens_per_document = tokenizer.num_start_tokens

    rnd = np.random if rng is None else rng
    if drop_first:
        proteins = proteins[1:]

    if shuffle:
        perm = rnd.permutation(len(proteins))
        if keep_first:
            perm = np.concatenate(([0], perm[perm != 0]))
    else:
        perm = np.arange(len(proteins))

    total_length = extra_tokens_per_document
    sampled_protein_ids = []
    sampled_protein_sequences = []
    sampled_protein_sequence_similarities = (
        [] if proteins.sequence_similarities is not None else None
    )
    sampled_protein_coverages = [] if proteins.coverages is not None else None
    sampled_protein_sequence_weights = (
        [] if proteins.sequence_weights is not None else None
    )

    for ix in perm:
        seq, pos, is_match = sequence_converter(proteins.sequences[ix])
        seq_length = len(seq) + extra_tokens_per_protein

        if max_tokens is not None and (total_length + seq_length >= max_tokens):
            leftover_tokens = (
                max_tokens - total_length - extra_tokens_per_protein
            )  # -1 for sep token
            leftover_tokens = min(leftover_tokens, leftover_tokens)
            if leftover_tokens > 0:
                if allow_partial_sequence:
                    seq_slice = _get_truncated_slice(len(seq), leftover_tokens, rnd)
                    sampled_protein_ids.append(ix)
                    sampled_protein_sequences.append(seq[seq_slice])
                    if proteins.sequence_similarities is not None:
                        sampled_protein_sequence_similarities.append(
                            proteins.sequence_similarities[ix]
                        )
                    if proteins.coverages is not None:
                        sampled_protein_coverages.append(proteins.coverages[ix])
                    if proteins.sequence_weights is not None:
                        sampled_protein_sequence_weights.append(
                            proteins.sequence_weights[ix]
                        )
                    total_length += len(seq[seq_slice]) + extra_tokens_per_protein
            break
        else:
            total_length += seq_length
            sampled_protein_ids.append(ix)
            sampled_protein_sequences.append(seq)
            if proteins.sequence_similarities is not None:
                sampled_protein_sequence_similarities.append(
                    proteins.sequence_similarities[ix]
                )
            if proteins.coverages is not None:
                sampled_protein_coverages.append(proteins.coverages[ix])
            if proteins.sequence_weights is not None:
                sampled_protein_sequence_weights.append(proteins.sequence_weights[ix])
    if len(sampled_protein_ids) == 0:
        raise ValueError("No proteins sampled: adjust max_tokens")
    # init will check array sizes - but misalignment could still occur
    return proteins[sampled_protein_ids].clone(
        sequences=sampled_protein_sequences,
        sequence_similarities=sampled_protein_sequence_similarities,
        coverages=sampled_protein_coverages,
        sequence_weights=sampled_protein_sequence_weights,
    )


def prepare_aligned_sequences_no_sampling(
    proteins: ProteinDocument,
    tokenizer: ProFamTokenizer,
    sequence_converter: Callable,
    **kwargs,
) -> ProteinDocument:
    """
    Prepare aligned sequences without subsampling/truncation.

    Applies the provided sequence_converter to each aligned sequence to
    obtain standardised sequence text and residue positions (alignment-aware),
    keeping all sequences.
    """
    converted_sequences = []
    for seq in proteins.sequences:
        new_seq, pos, _ = sequence_converter(seq)
        converted_sequences.append(new_seq)
    return proteins.clone(sequences=converted_sequences)


def filter_by_length(
    proteins: ProteinDocument,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    **kwargs,
):
    if min_length is None and max_length is None:
        return proteins
    else:

        def length_filter(protein: Protein):
            assert not "[" in protein.sequence
            return (min_length is None or len(protein.sequence) >= min_length) and (
                max_length is None or len(protein.sequence) <= max_length
            )

        return proteins.filter(length_filter)


def replace_selenocysteine_pyrrolysine(proteins: ProteinDocument, **kwargs):
    new_sequences = [
        seq.replace("U", "C").replace("O", "K") for seq in proteins.sequences
    ]
    return proteins.clone(sequences=new_sequences)


def add_final_sep(proteins: ProteinDocument, tokenizer: ProFamTokenizer, **kwargs):
    """Add a separator token to the end of the last sequence and extend other arrays accordingly.

    Args:
        proteins: ProteinDocument containing the proteins to modify
        tokenizer: ProFamTokenizer containing the separator token

    Returns:
        Modified ProteinDocument with separator token added only to the last protein
    """
    # Add sep token only to the last sequence
    new_sequences = []
    for i, seq in enumerate(proteins.sequences):
        if i == len(proteins.sequences) - 1:  # Last protein
            new_sequences.append(seq + tokenizer.sep_token)
        else:
            new_sequences.append(seq)

    return proteins.clone(
        sequences=new_sequences,
    )


def random_crop(
    proteins: ProteinDocument,
    min_length: int,
    max_length: int,
    crop_prob: float = 0.5,
    rng: Optional[np.random.Generator] = None,
    **kwargs,
):
    assert len(proteins) == 1, "random_crop only supports single protein documents"
    if rng is None:
        rng = np.random
    if rng.random() > crop_prob:
        return proteins
    elif len(proteins.sequences[0]) < min_length:
        return proteins
    else:
        length = rng.randint(
            min_length, min(max_length + 1, len(proteins.sequences[0]) + 1)
        )
        start = rng.randint(0, len(proteins.sequences[0]) - length + 1)
        end = start + length
        return proteins.slice_arrays([slice(start, end)])


def apply_transforms(
    transforms,
    proteins,
    tokenizer,
    max_tokens: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
):
    for transform in transforms or []:
        proteins = transform(
            proteins, tokenizer=tokenizer, max_tokens=max_tokens, rng=rng
        )
    return proteins


AAs = "ACDEFGHIKLMNPQRSTVWY"
