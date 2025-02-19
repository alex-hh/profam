from typing import Callable, List, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from src.constants import BACKBONE_ATOMS
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
    truncated_sequence_lengths = np.minimum(
        new_sequence_lengths, tokenizer.max_res_pos_in_seq or max_length
    )
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
        if tokenizer.max_res_pos_in_seq is not None:
            array_slices = [
                _get_truncated_slice(
                    new_sequence_lengths[i] - extra_tokens_per_protein,
                    tokenizer.max_res_pos_in_seq,
                    rnd,
                )
                for i in range(effective_endpoint)
            ]
        else:
            array_slices = [None] * effective_endpoint

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
        if tokenizer.max_res_pos_in_seq is not None:
            array_slices = [
                _get_truncated_slice(
                    new_sequence_lengths[i] - extra_tokens_per_protein,
                    tokenizer.max_res_pos_in_seq,
                    rnd,
                )
                for i in range(len(new_proteins))
            ]
        else:
            array_slices = [None] * len(new_proteins)

    new_proteins = new_proteins.clone(
        residue_positions=[
            list(range(1, len(seq) + 1)) for seq in new_proteins.sequences
        ]
    )
    new_proteins = new_proteins.slice_arrays(array_slices)
    total_tokens = (
        sum(len(seq) for seq in new_proteins.sequences)
        + extra_tokens_per_protein * len(new_proteins)
        + extra_tokens_per_document
    )
    if total_tokens > max_tokens:
        bp = 1
        raise ValueError(
            f"Total tokens {total_tokens} > max_tokens {max_tokens}"
            f"in {proteins.identifier}"
        )
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
    sampled_protein_positions = []

    for ix in perm:
        seq, pos, is_match = sequence_converter(proteins.sequences[ix])
        if any(
            attr is not None
            for attr in [
                proteins.backbone_coords,
                proteins.structure_tokens,
                proteins.plddts,
            ]
        ):
            raise NotImplementedError(
                "To handle structure alignment we require something more sophisticated"
            )
        seq_length = len(seq) + extra_tokens_per_protein

        if max_tokens is not None and (total_length + seq_length >= max_tokens):
            leftover_tokens = (
                max_tokens - total_length - extra_tokens_per_protein
            )  # -1 for sep token
            leftover_tokens = min(
                leftover_tokens, tokenizer.max_res_pos_in_seq or leftover_tokens
            )
            if leftover_tokens > 0:
                seq_slice = _get_truncated_slice(len(seq), leftover_tokens, rnd)
                sampled_protein_ids.append(ix)
                sampled_protein_sequences.append(seq[seq_slice])
                sampled_protein_positions.append(pos[seq_slice])
                total_length += len(seq[seq_slice]) + extra_tokens_per_protein
            break
        elif (
            tokenizer.max_res_pos_in_seq is not None
            and seq_length > tokenizer.max_res_pos_in_seq
        ):
            # N.B. assumes no addition or removal of residues in sequence conversion
            seq_slice = _get_truncated_slice(
                len(seq), tokenizer.max_res_pos_in_seq, rnd
            )
            sampled_protein_ids.append(ix)
            sampled_protein_sequences.append(seq[seq_slice])
            sampled_protein_positions.append(pos[seq_slice])
            total_length += len(seq[seq_slice]) + extra_tokens_per_protein
        else:
            total_length += seq_length
            sampled_protein_ids.append(ix)
            sampled_protein_sequences.append(seq)
            sampled_protein_positions.append(pos)
    if len(sampled_protein_ids) == 0:
        raise ValueError("No proteins sampled: adjust max_tokens")
    # init will check array sizes - but misalignment could still occur
    return proteins[sampled_protein_ids].clone(
        residue_positions=sampled_protein_positions,
        sequences=sampled_protein_sequences,
    )


def noise_backbones(
    proteins: ProteinDocument,
    std: float = 0.1,
    rng: Optional[np.random.Generator] = None,
    **kwargs,
):
    """Add Gaussian noise to the backbone coordinates.

    ProteinMPNN:
    We found that training models on backbones to which Gaussian
    noise (std=0.02Å) had been added improved sequence recovery on confident
    protein structure models generated by AlphaFold (average pLDDT>80.0) from UniRef50,
    (46.9->48.5) while the sequence recovery on unperturbed PDB structures significantly
    decreased (50.8->47.9); crystallographic refinement may impart some memory of
    amino acid identity in the backbone coordinates which is captured by models
    trained on crystal structure backbones and reduced by the addition of noise.
    Robustness to small displacements in atomic coordinates is a
    desirable feature in real world applications where the protein
    backbone geometry is not known at atomic resolution.

    LigandMPNN:
    Protein and context atoms were noised by adding 0.1 A standard deviation
    Gaussian noise to avoid protein backbone memorization

    ESM-IF: Even after Amber relaxation, the backbone coordinates predicted
    by AlphaFold2 contain artifacts in the sub-Angstrom scale that may give away amino
    acid identities. Without adding noise on predicted structures, there is a
    substantial gap between held-out set performance on predicted structures and
    on experimental structures. To prevent the model from learning non-generalizable
    AlphaFold2-specific rules, we added Gaussian noise at the 0.1A scale on predicted
    backbone coordinates. The Gaussian noise improves the invariant Transformer
    performance (preplexity on CATH proteins 4.32->4.10 but not the GVP-GNN performance)
    """
    new_coords = []
    for coords in proteins.backbone_coords:
        assert coords.ndim == 3  # l, 4, 3
        noise = np.random.normal(scale=std, size=coords.shape)
        new_coords.append(coords + noise)
    return proteins.clone(backbone_coords=new_coords)


def mask_atoms(proteins: ProteinDocument, atom_names: Optional[List] = None, **kwargs):
    if atom_names is None:
        atom_names = BACKBONE_ATOMS
    new_coords = []
    new_coords_masks = []
    for coords, coords_mask in zip(
        proteins.backbone_coords, proteins.backbone_coords_masks
    ):
        assert coords.ndim == 3
        assert coords_mask.ndim == 3
        atom_ids = [BACKBONE_ATOMS.index(at) for at in atom_names]
        coords[:, atom_ids, :] = np.nan
        coords_mask[:, atom_ids, :] = 0.0
        new_coords.append(coords)
        new_coords_masks.append(coords_mask)
    return proteins.clone(
        backbone_coords=new_coords, backbone_coords_masks=new_coords_masks
    )


def rescale_backbones(proteins: ProteinDocument, scale: float = 6.0, **kwargs):
    # AF3 has a time-dependent scale (constant variance at all timesteps). They use 4 for t -0
    # We can use a fixed scale for now.
    new_coords = []
    for coords in proteins.backbone_coords:
        assert coords.ndim == 3  # l, 4, 3
        new_coords.append(coords / scale)
    return proteins.clone(backbone_coords=new_coords)


def rotate_backbones(
    proteins: ProteinDocument, rng: Optional[np.random.Generator] = None, **kwargs
):
    new_coords = []
    for coords in proteins.backbone_coords:
        # apply a separate random rotation to each protein
        # TODO: handle nans.
        assert coords.ndim == 3  # l, 4, 3
        if np.isnan(coords).all():
            new_coords.append(coords)
        else:
            rotation = R.random(random_state=rng)
            flat_coords = coords.reshape(-1, 3)
            flat_nan_mask = np.isnan(flat_coords).any(axis=1)
            flat_coords[~flat_nan_mask] = rotation.apply(flat_coords[~flat_nan_mask])
            flat_coords = flat_coords.reshape(-1, 4, 3)
            new_coords.append(flat_coords)
    return proteins.clone(backbone_coords=new_coords)


def centre_backbones(proteins: ProteinDocument, **kwargs):
    """Centres the coordinates, so that the centroid (average position) of the backbone atoms is at the origin.
    AF3 centres and then randomly translates (Alg 19.)
    """
    # TODO: handle nans.
    new_coords = []
    for coords in proteins.backbone_coords:
        assert coords.ndim == 3  # l, 4, 3
        if np.isnan(coords).all():
            new_coords.append(coords)
        else:
            centroid = np.nanmean(coords)
            new_coords.append(coords - centroid)
    return proteins.clone(backbone_coords=new_coords)


def replace_nans_in_coords(
    proteins: ProteinDocument, fill_value: float = 0.0, **kwargs
):
    # n.b. this should occur after any nan-aware transforms like centering, roation.
    new_coords = []
    for coords in proteins.backbone_coords:
        assert coords.ndim == 3  # l, 4, 3
        new_coords.append(np.nan_to_num(coords, nan=fill_value))
    return proteins.clone(backbone_coords=new_coords)


def fill_missing_fields(
    proteins: ProteinDocument, tokenizer: ProFamTokenizer, **kwargs
):
    if not proteins.has_all_structure_arrays:
        proteins = proteins.fill_missing_structure_arrays(
            coords_fill=np.nan,
            plddts_fill=100.0,
            tokens_fill=tokenizer.mask_token,
        )
    return proteins


def apply_plddt_mask(
    proteins: ProteinDocument,
    tokenizer: ProFamTokenizer,
    threshold: float = 80.0,
    mask_plddts: bool = False,
    mask_sequences: bool = False,
    **kwargs,
):
    # only mask structure tokens
    # must be before replace nans and before interleaving
    masked_coords = []
    masked_coords_masks = []
    masked_sequences = []
    masked_structure_tokens = []
    masked_plddts = []
    assert (
        proteins.interleaved_coords_masks is None
    ), "plddt masking should be applied before interleaving"
    for ix, (sequence, coords, coords_mask, plddts) in enumerate(
        zip(
            proteins.sequences,
            proteins.backbone_coords,
            proteins.backbone_coords_masks,
            proteins.plddts,
        )
    ):
        plddt_mask = plddts < threshold
        masked_coords.append(np.where(plddt_mask[:, None, None], np.nan, coords))
        masked_coords_masks.append(
            np.where(plddt_mask[:, None, None], 0.0, coords_mask)
        )
        if proteins.structure_tokens is not None:
            structure_tokens = proteins.structure_tokens[ix]
            masked_structure_tokens.append(
                "".join(
                    [
                        tok if not m else tokenizer.mask_token
                        for tok, m in zip(structure_tokens, plddt_mask)
                    ]
                )
            )
        if mask_sequences:
            masked_sequences.append(
                "".join(
                    [
                        aa if not m else tokenizer.mask_token
                        for aa, m in zip(sequence, plddt_mask)
                    ]
                )
            )
        else:
            masked_sequences.append(sequence)
        if mask_plddts:
            masked_plddts.append(np.where(plddt_mask, 0.0, plddts))
        else:
            masked_plddts.append(plddts)

    return proteins.clone(
        sequences=masked_sequences,
        structure_tokens=masked_structure_tokens if masked_structure_tokens else None,
        backbone_coords=masked_coords,
        backbone_coords_masks=masked_coords_masks,
        plddts=masked_plddts,
    )


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


def interleave_structure_sequence(
    proteins: ProteinDocument,
    tokenizer: ProFamTokenizer,
    structure_first_prob: float = 1.0,
    max_tokens: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    repeat_coords: bool = False,  # in first runs we used repeat coords - this requires modified sampling code.
):
    """Automatically reduces the number of proteinss to fit within max_tokens.

    N.B. we hard-code coords padding as 0.
    """
    rnd = np.random if rng is None else rng
    coin_flip = rnd.rand()
    interleaved_sequences = []
    interleaved_positions = []
    interleaved_plddts = []
    interleaved_coords = []
    interleaved_structure_coords_masks = []
    interleaved_sequence_coords_masks = []
    interleaved_modality_masks = []
    total_tokens = tokenizer.num_start_tokens
    for ix, seq in enumerate(proteins.sequences):
        if proteins.structure_tokens is not None:
            seq_3d = proteins.structure_tokens[ix]
        else:
            seq_3d = tokenizer.mask_token_id * len(seq)
        if proteins.backbone_coords is not None:
            xyz = proteins.backbone_coords[ix]
            coords_mask = proteins.backbone_coords_masks[ix]
        else:
            xyz = np.zeros((len(seq), 4, 3))
            coords_mask = np.zeros((len(seq), 4, 3))
        if proteins.plddts is not None:
            plddts = proteins.plddts[ix]
        else:
            plddts = np.full((len(seq),), 100.0)
        positions = proteins.residue_positions[ix]
        # TODO: monitor max_tokens
        assert (
            len(seq) == len(xyz) == len(plddts)
        ), f"seq {seq} length != xyz shape {xyz.shape[0]} or plddts {plddts.shape[0]}"  # n.b. special tokens can screw this up
        assert isinstance(positions, list)
        if coin_flip < structure_first_prob:
            interleaved_sequences.append(seq_3d + tokenizer.seq_struct_sep_token + seq)
            interleaved_positions.append(
                positions + [-1] + positions
            )  # 1 will be added to positions in tokenizer so we use -1
            interleaved_plddts.append(
                np.concatenate(
                    [np.array(plddts), np.full((1,), 100.0), np.array(plddts)]
                )
            )
            interleaved_coords.append(
                np.concatenate(
                    [
                        xyz,
                        np.full((1, 4, 3), 0.0),
                        xyz if repeat_coords else np.zeros_like(xyz),
                    ],
                    axis=0,
                )
            )
            interleaved_structure_coords_masks.append(
                np.concatenate(
                    [coords_mask, np.zeros((1, 4, 3)), np.zeros_like(xyz)], axis=0
                )
            )
            interleaved_sequence_coords_masks.append(
                np.concatenate(
                    [np.zeros_like(xyz), np.zeros((1, 4, 3)), coords_mask], axis=0
                )
            )
            sequence_mask = np.concatenate(
                [np.zeros((xyz.shape[0] + 1,)), np.ones((xyz.shape[0],))]
            )
            structure_mask = np.concatenate(
                [np.ones((xyz.shape[0],)), np.zeros((xyz.shape[0] + 1,))]
            )
            interleaved_modality_masks.append(
                np.stack([sequence_mask, structure_mask], axis=-1).astype(bool)
            )
        else:
            interleaved_sequences.append(seq + tokenizer.seq_struct_sep_token + seq_3d)
            interleaved_positions.append(positions + [-1] + positions)
            interleaved_plddts.append(
                np.concatenate(
                    [np.array(plddts), np.full((1,), 100.0), np.array(plddts)]
                )
            )
            interleaved_coords.append(
                np.concatenate(
                    [np.zeros_like(xyz), np.full((1, 4, 3), 0.0), xyz], axis=0
                )
            )
            interleaved_structure_coords_masks.append(
                np.concatenate(
                    [np.zeros_like(xyz), np.zeros((1, 4, 3)), coords_mask], axis=0
                )
            )
            interleaved_sequence_coords_masks.append(
                np.concatenate(
                    [coords_mask, np.zeros((1, 4, 3)), np.zeros_like(xyz)], axis=0
                )
            )
            sequence_mask = np.concatenate(
                [np.ones((xyz.shape[0],)), np.zeros((xyz.shape[0] + 1,))]
            )
            structure_mask = np.concatenate(
                [np.zeros((xyz.shape[0] + 1,)), np.ones((xyz.shape[0],))]
            )
            interleaved_modality_masks.append(
                np.stack([sequence_mask, structure_mask], axis=-1).astype(bool)
            )

        assert not "[" in seq
        total_tokens += len(seq) * 2 + 2  # +1 for each separator

        if total_tokens > (max_tokens or 1e8):
            interleaved_sequences = interleaved_sequences[:-1]
            interleaved_positions = interleaved_positions[:-1]
            interleaved_plddts = interleaved_plddts[:-1]
            interleaved_coords = interleaved_coords[:-1]
            interleaved_structure_coords_masks = interleaved_structure_coords_masks[:-1]
            interleaved_sequence_coords_masks = interleaved_sequence_coords_masks[:-1]
            interleaved_modality_masks = interleaved_modality_masks[:-1]
            assert (
                len(interleaved_sequences) > 0
            ), f"Cannot fit any sequences in max_tokens sequence length {len(seq)} (becomes {len(seq) * 2 + 2}) max {max_tokens}"
            break

    return proteins.clone(
        sequences=interleaved_sequences,
        residue_positions=interleaved_positions,
        plddts=interleaved_plddts,
        backbone_coords=interleaved_coords,
        backbone_coords_masks=interleaved_structure_coords_masks,
        interleaved_coords_masks=interleaved_sequence_coords_masks,
        modality_masks=interleaved_modality_masks,
        structure_tokens=None,
    )


def replace_selenocysteine_pyrrolysine(proteins: ProteinDocument, **kwargs):
    new_sequences = [
        seq.replace("U", "C").replace("O", "K") for seq in proteins.sequences
    ]
    return proteins.clone(sequences=new_sequences)


def add_final_sep(proteins: ProteinDocument, tokenizer: ProFamTokenizer, **kwargs):
    """Add a separator token to the end of each sequence and extend other arrays accordingly.

    Args:
        proteins: ProteinDocument containing the proteins to modify
        tokenizer: ProFamTokenizer containing the separator token

    Returns:
        Modified ProteinDocument with separator tokens added and arrays extended
    """
    new_sequences = [seq + tokenizer.sep_token for seq in proteins.sequences]

    # Handle residue positions - add -1 for separator (similar to interleave_structure_sequence)
    new_positions = (
        [pos + [-1] for pos in proteins.residue_positions]
        if proteins.residue_positions is not None
        else None
    )

    # Handle plddts - extend with 100.0 (full confidence for special tokens)
    new_plddts = (
        [np.concatenate([plddt, np.array([100.0])]) for plddt in proteins.plddts]
        if proteins.plddts is not None
        else None
    )

    # Handle backbone coordinates - extend with zeros
    new_coords = (
        [
            np.concatenate([coords, np.zeros((1, 4, 3))], axis=0)
            for coords in proteins.backbone_coords
        ]
        if proteins.backbone_coords is not None
        else None
    )

    # Handle backbone coordinate masks - extend with zeros
    new_coords_masks = (
        [
            np.concatenate([mask, np.zeros((1, 4, 3))], axis=0)
            for mask in proteins.backbone_coords_masks
        ]
        if proteins.backbone_coords_masks is not None
        else None
    )

    # Handle structure tokens - extend with mask token
    new_structure_tokens = (
        [tokens + tokenizer.mask_token for tokens in proteins.structure_tokens]
        if proteins.structure_tokens is not None
        else None
    )

    # Handle modality masks if present - extend with zeros for sequence position
    new_modality_masks = (
        [
            np.concatenate(
                [mask, np.array([[1, 0]])], axis=0
            )  # [1,0] indicates sequence token
            for mask in proteins.modality_masks
        ]
        if proteins.modality_masks is not None
        else None
    )

    return proteins.clone(
        sequences=new_sequences,
        residue_positions=new_positions,
        plddts=new_plddts,
        backbone_coords=new_coords,
        backbone_coords_masks=new_coords_masks,
        structure_tokens=new_structure_tokens,
        modality_masks=new_modality_masks,
    )


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


def repeat_random_sequence_in_family(
    proteins: ProteinDocument,
    tokenizer: ProFamTokenizer,
    max_tokens: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    **kwargs,
) -> ProteinDocument:
    """
    FOR DEBUGGING / BENCHMARKING ONLY
    Replaces all sequences in the ProteinDocument with a randomly selected one from the family.
    """
    rng = kwargs.get("rng", np.random.default_rng())
    index = rng.choice(len(proteins.sequences))
    proteins.sequences = [proteins.sequences[index]] * len(proteins.sequences)
    # If residue_positions are present, repeat them as well
    if proteins.residue_positions is not None:
        positions = proteins.residue_positions[index]
        proteins.residue_positions = [positions] * len(proteins.sequences)
    if proteins.backbone_coords is not None:
        coords = proteins.backbone_coords[index]
        proteins.backbone_coords = [coords] * len(proteins.sequences)
    if proteins.backbone_coords_masks is not None:
        coords_masks = proteins.backbone_coords_masks[index]
        proteins.backbone_coords_masks = [coords_masks] * len(proteins.sequences)
    if proteins.plddts is not None:
        plddts = proteins.plddts[index]
        proteins.plddts = [plddts] * len(proteins.sequences)
    if proteins.modality_masks is not None:
        modality_masks = proteins.modality_masks[index]
        proteins.modality_masks = [modality_masks] * len(proteins.sequences)
    if proteins.interleaved_coords_masks is not None:
        interleaved_coords_masks = proteins.interleaved_coords_masks[index]
        proteins.interleaved_coords_masks = [interleaved_coords_masks] * len(
            proteins.sequences
        )
    if proteins.structure_tokens is not None:
        structure_tokens = proteins.structure_tokens[index]
        proteins.structure_tokens = [structure_tokens] * len(proteins.sequences)
    if proteins.accessions is not None:
        accessions = proteins.accessions[index]
        proteins.accessions = [accessions] * len(proteins.sequences)
    if max_tokens is not None:
        proteins = preprocess_raw_sequences_sampling_to_max_tokens(
            proteins=proteins,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
        )
    return proteins


def seq_is_random_res_pos(
    proteins: ProteinDocument, rng: Optional[np.random.Generator] = None, **kwargs
) -> ProteinDocument:
    """
    FOR DEBUGGING / BENCHMARKING ONLY
    Replaces residue indices with random ones (1-20)
    then makes sequence from AA letters indexed by the random indices
    """
    rng = kwargs.get("rng", np.random.default_rng())
    new_sequences = []
    residue_positions = []
    AAs = "ACDEFGHIKLMNPQRSTVWY"
    for seq in proteins.sequences:
        random_indices = rng.choice(20, size=len(seq))
        residue_positions.append(random_indices)
        new_sequences.append("A" + "".join([AAs[i] for i in random_indices[:-1]]))
    return proteins.clone(sequences=new_sequences, residue_positions=residue_positions)
