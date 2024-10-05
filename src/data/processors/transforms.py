import functools
import itertools
from typing import Callable, List, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from src.constants import BACKBONE_ATOMS
from src.data.objects import Protein, ProteinDocument
from src.data.tokenizers import ProFamTokenizer
from src.sequence.fasta import convert_sequence_with_positions
from src.utils.utils import np_random

from .preprocessing import PreprocessingConfig


def convert_aligned_sequence_adding_positions(
    seq,
    keep_gaps=True,
    keep_insertions=True,
    to_upper=False,
    use_msa_pos: bool = True,
):
    """
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


def preprocess_sequences(
    proteins: ProteinDocument,
    tokenizer: ProFamTokenizer,
    sequence_converter: Callable = convert_raw_sequence_adding_positions,
    **kwargs,
) -> ProteinDocument:
    sequences, positions = [], []
    for seq in proteins.sequences:
        seq, pos, is_match = sequence_converter(seq)
        sequences.append(seq)
        positions.append(pos)
    return proteins.clone(sequences=sequences, positions=positions)


def preprocess_sequences_sampling_to_max_tokens(
    proteins: ProteinDocument,
    tokenizer: ProFamTokenizer,
    max_tokens: Optional[int] = None,
    shuffle: bool = True,
    drop_first: bool = False,
    keep_first: bool = False,
    seed: Optional[int] = None,
    sequence_converter: Callable = convert_raw_sequence_adding_positions,
    **kwargs,
) -> ProteinDocument:
    """
    Sample proteins to fit within a maximum token limit.

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

    if drop_first:
        proteins = proteins[1:]

    if shuffle:
        rnd = np_random(seed)
        perm = rnd.permutation(len(proteins))
        if keep_first:
            perm = np.concatenate(([0], perm[perm != 0]))
    else:
        perm = range(len(proteins))

    if max_tokens is None:
        return proteins[perm]

    total_length = extra_tokens_per_document
    sampled_protein_ids = []
    sampled_protein_sequences = []
    sampled_protein_positions = []
    for ix in perm:
        seq, pos, is_match = sequence_converter(proteins.sequences[ix])
        seq_length = len(seq) + extra_tokens_per_protein
        if total_length + seq_length > max_tokens:
            break
        total_length += seq_length
        sampled_protein_ids.append(ix)
        sampled_protein_sequences.append(seq)
        sampled_protein_positions.append(pos)

    return proteins[sampled_protein_ids].clone(
        positions=sampled_protein_positions,
        sequences=sampled_protein_sequences,
    )


def noise_backbones(proteins: ProteinDocument, std: float = 0.1, **kwargs):
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


def rotate_backbones(proteins: ProteinDocument, **kwargs):
    new_coords = []
    for coords in proteins.backbone_coords:
        # apply a separate random rotation to each protein
        # TODO: handle nans.
        assert coords.ndim == 3  # l, 4, 3
        if np.isnan(coords).all():
            new_coords.append(coords)
        else:
            rotation = R.random()
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
    repeat_coords: bool = False,  # in first runs we used repeat coords - this requires modified sampling code.
):
    """Automatically reduces the number of proteinss to fit within max_tokens.

    N.B. we hard-code coords padding as 0.
    """
    coin_flip = np.random.rand()
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
        positions = proteins.positions[ix]
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
            ), f"Cannot fit any sequences in max_tokens tried {total_tokens} max {max_tokens}"
            break

    return proteins.clone(
        sequences=interleaved_sequences,
        positions=interleaved_positions,
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


def apply_transforms(transforms, proteins, tokenizer, max_tokens: Optional[int] = None):
    for transform in transforms or []:
        proteins = transform(proteins, tokenizer=tokenizer, max_tokens=max_tokens)
    return proteins
