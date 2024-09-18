import bisect
import itertools
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from src.data.fasta import convert_sequence_with_positions
from src.data.objects import Protein, ProteinDocument
from src.utils.tokenizers import ProFamTokenizer
from src.utils.utils import np_random


def convert_sequences_adding_positions(
    proteins: ProteinDocument,
    keep_gaps: bool = False,
    keep_insertions: bool = True,
    to_upper: bool = True,
    use_msa_pos: bool = False,
    truncate_after_n_sequences: Optional[int] = None,
    **kwargs,
):
    sequences = []
    positions = []
    for seq in itertools.islice(proteins.sequences, truncate_after_n_sequences):
        seq, pos, _ = convert_sequence_with_positions(
            seq,
            keep_gaps=keep_gaps,
            keep_insertions=keep_insertions,
            to_upper=to_upper,
            use_msa_pos=use_msa_pos,
        )
        sequences.append(seq)
        positions.append(pos)
    return proteins.clone(
        sequences=sequences,
        positions=positions,
    )


# TODO: implement rotation, centering, scaling of coordinates
def sample_to_max_tokens(
    proteins: ProteinDocument,
    max_tokens: int,
    tokenizer: Optional[ProFamTokenizer] = None,
    shuffle: bool = True,
    seed: Optional[int] = None,
    drop_first: bool = False,
    keep_first: bool = False,
    extra_tokens_per_document: Optional[int] = None,
    **kwargs,
):
    extra_tokens_per_sequence = 1  # for separator. TODO infer from tokenizer?
    if tokenizer is not None:
        extra_tokens_per_document = tokenizer.num_start_tokens
    else:
        assert extra_tokens_per_document is not None
        extra_tokens_per_document = 2
    # extra_arrays = [positions, proteins.coords, proteins.plddts, proteins.structure_tokens]
    rnd = np_random(seed)
    if drop_first:
        proteins = proteins[1:]
    if shuffle:
        perm = rnd.permutation(len(proteins))
        if keep_first:
            perm[0] = 0
        proteins = proteins[perm]

    if max_tokens is not None:
        cumulative_lengths = list(
            itertools.accumulate(
                [len(s) + extra_tokens_per_sequence for s in proteins.sequences]
            )
        )
        insertion_point = bisect.bisect_left(
            cumulative_lengths,
            max_tokens - extra_tokens_per_document,
        )
    else:
        insertion_point = len(proteins)
    proteins = proteins[:insertion_point]
    return proteins


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


def fill_missing_fields(proteins: ProteinDocument, tokenizer: ProFamTokenizer):
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
):
    # only mask structure tokens
    # must be before replace nans and before interleaving
    masked_coords = []
    masked_coords_masks = []
    masked_sequences = []
    assert (
        proteins.interleaved_coords_masks is None
    ), "plddt masking should be applied before interleaving"
    for sequence, coords, coords_mask, plddts in zip(
        proteins.sequences,
        proteins.backbone_coords,
        proteins.backbone_coords_masks,
        proteins.plddts,
    ):
        plddt_mask = plddts < threshold
        masked_coords.append(np.where(plddt_mask[:, None, None], np.nan, coords))
        masked_coords_masks.append(
            np.where(plddt_mask[:, None, None], 0.0, coords_mask)
        )
        masked_sequences.append(
            "".join(
                [
                    aa if not m else tokenizer.mask_token
                    for aa, m in zip(sequence, plddt_mask)
                ]
            )
        )

    return proteins.clone(
        sequences=masked_sequences,
        backbone_coords=masked_coords,
        backbone_coords_masks=masked_coords_masks,
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
    # TODO: allow arbitrary mask probability generating functions
    conditioning_sequence_mask_prob: tuple = (
        1.0,
        1.0,
    ),  # range for uniform sampling of probability
    conditioning_structure_mask_prob: tuple = (1.0, 1.0),
    remove_conditioning_sequence_from_targets: bool = False,
    remove_conditioning_structure_from_targets: bool = False,
    use_structure_tokens: bool = True,
    # TODO: repeat_coords, repeat_sequence kwargs
):
    """Interleave structure and sequence tokens.

    This transform allows structure-conditioned sequence generation and sequence-conditioned structure generation.

    To increase flexibility, we allow conditioning on known amino acids during structure-conditioned
    sequence generation and known structure during sequence-conditioned structure generation.
    This is done by masking the target modality at positions belonging to the conditioning modality
    according to the probabilities.

    This basically allows an MAR-style model (Kaiming He).

    N.B. we could randomise the order of the generated tokens -
    but we already support a discrete-diffusion style parallel proposal generation
    even without doing this.

    The function automatically reduces the number of proteins to fit within max_tokens.

    TODO: make more readable somehow

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

    # verify_mask_probs(conditioning_sequence_mask_prob, conditioning_structure_mask_prob)
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

            targets_plddts = np.array(plddts)
            targets_coords = np.zeros_like(xyz)
            targets_coords_mask = np.zeros_like(coords_mask)
            targets_interleaved_sequence_coords_mask = coords_mask
            targets_sequence_mask = np.ones((xyz.shape[0],))
            targets_structure_mask = np.zeros((xyz.shape[0],))
            targets_positions = positions

            if use_structure_tokens:
                assert conditioning_sequence_mask_prob[0] == 1.0
                assert not remove_conditioning_sequence_from_targets
                interleaved_seq = seq_3d + tokenizer.seq_struct_sep_token + seq
            else:
                mask_frac = np.random.uniform(*conditioning_sequence_mask_prob)
                mask = np.random.rand(len(seq)) < mask_frac  # positions to be predicted
                masked_seq = "".join(
                    [aa if not m else tokenizer.mask_token for aa, m in zip(seq, mask)]
                )
                if remove_conditioning_sequence_from_targets:
                    targets_seq = [aa for aa, m in zip(seq, mask) if m]
                    targets_positions = [p for p, m in zip(positions, mask) if m]
                    targets_plddts = targets_plddts[mask]
                    targets_coords = targets_coords[mask]
                    targets_coords_mask = targets_coords_mask[mask]
                    targets_interleaved_sequence_coords_mask = (
                        targets_interleaved_sequence_coords_mask[mask]
                    )
                    targets_sequence_mask = targets_sequence_mask[mask]
                    targets_structure_mask = targets_structure_mask[mask]
                else:
                    targets_seq = seq
                interleaved_seq = (
                    masked_seq + tokenizer.seq_struct_sep_token + targets_seq
                )

            interleaved_sequences.append(interleaved_seq)
            interleaved_positions.append(
                positions + [-1] + targets_positions
            )  # 1 will be added to positions in tokenizer so we use -1
            interleaved_plddts.append(
                np.concatenate([np.array(plddts), np.full((1,), 100.0), targets_plddts])
            )
            interleaved_coords.append(
                np.concatenate([xyz, np.full((1, 4, 3), 0.0), targets_coords], axis=0)
            )
            interleaved_structure_coords_masks.append(
                np.concatenate(
                    [coords_mask, np.zeros((1, 4, 3)), targets_coords_mask], axis=0
                )
            )
            interleaved_sequence_coords_masks.append(
                np.concatenate(
                    [
                        np.zeros_like(xyz),
                        np.zeros((1, 4, 3)),
                        targets_interleaved_sequence_coords_mask,
                    ],
                    axis=0,
                )
            )
            sequence_mask = np.concatenate(
                [np.zeros((xyz.shape[0] + 1,)), targets_sequence_mask]
            )
            structure_mask = np.concatenate(
                [np.ones((xyz.shape[0],)), np.zeros((1,)), targets_structure_mask]
            )
            interleaved_modality_masks.append(
                np.stack([sequence_mask, structure_mask], axis=-1).astype(bool)
            )

        else:

            targets_plddts = np.array(plddts)
            targets_coords = xyz
            targets_coords_mask = coords_mask
            targets_interleaved_sequence_coords_mask = np.zeros_like(coords_mask)
            targets_sequence_mask = np.zeros((xyz.shape[0],))
            targets_structure_mask = np.ones((xyz.shape[0],))
            targets_positions = positions

            if use_structure_tokens:
                assert (
                    conditioning_structure_mask_prob[0] == 1.0
                ), "We can't partially mask structure tokens"
                interleaved_seq = seq + tokenizer.seq_struct_sep_token + seq_3d
                masked_coords = np.zeros_like(xyz)
                masked_coords_mask = np.zeros_like(coords_mask)
            else:
                mask_frac = np.random.uniform(*conditioning_structure_mask_prob)
                mask = np.random.rand(len(seq)) < mask_frac  # positions to be predicted
                masked_coords = np.where(mask[:, None, None], xyz, 0.0)
                masked_coords_mask = np.where(mask[:, None, None], coords_mask, 0.0)

                if remove_conditioning_structure_from_targets:
                    targets_seq = tokenizer.mask_token * mask.sum()
                    targets_positions = [p for p, m in zip(positions, mask) if m]
                    targets_plddts = targets_plddts[mask]
                    targets_coords = targets_coords[mask]
                    targets_coords_mask = targets_coords_mask[mask]
                    targets_interleaved_sequence_coords_mask = (
                        targets_interleaved_sequence_coords_mask[mask]
                    )
                    targets_sequence_mask = targets_sequence_mask[mask]
                    targets_structure_mask = targets_structure_mask[mask]
                else:
                    targets_seq = tokenizer.mask_token * len(seq)
                interleaved_seq = seq + tokenizer.seq_struct_sep_token + targets_seq

            interleaved_sequences.append(interleaved_seq)
            interleaved_positions.append(positions + [-1] + targets_positions)
            interleaved_plddts.append(
                np.concatenate([np.array(plddts), np.full((1,), 100.0), targets_plddts])
            )
            interleaved_coords.append(
                np.concatenate(
                    [masked_coords, np.full((1, 4, 3), 0.0), targets_coords], axis=0
                )
            )
            interleaved_structure_coords_masks.append(
                np.concatenate(
                    [masked_coords_mask, np.zeros((1, 4, 3)), targets_coords_mask],
                    axis=0,
                )
            )
            interleaved_sequence_coords_masks.append(
                np.concatenate(
                    [
                        coords_mask,
                        np.zeros((1, 4, 3)),
                        targets_interleaved_sequence_coords_mask,
                    ],
                    axis=0,
                )
            )
            sequence_mask = np.concatenate(
                [np.ones((xyz.shape[0],)), np.zeros((1,)), targets_sequence_mask]
            )
            structure_mask = np.concatenate(
                [np.zeros((xyz.shape[0] + 1,)), targets_structure_mask]
            )
            interleaved_modality_masks.append(
                np.stack([sequence_mask, structure_mask], axis=-1).astype(bool)
            )

        assert not "[" in seq
        total_tokens += len(seq) * 2 + 2  # +1 for each separator

        if total_tokens > tokenizer.max_tokens:
            interleaved_sequences = interleaved_sequences[:-1]
            interleaved_positions = interleaved_positions[:-1]
            interleaved_plddts = interleaved_plddts[:-1]
            interleaved_coords = interleaved_coords[:-1]
            interleaved_structure_coords_masks = interleaved_structure_coords_masks[:-1]
            interleaved_sequence_coords_masks = interleaved_sequence_coords_masks[:-1]
            interleaved_modality_masks = interleaved_modality_masks[:-1]
            assert (
                len(interleaved_sequences) > 0
            ), f"Cannot fit any sequences in max_tokens tried {total_tokens} max {tokenizer.max_tokens}"
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


def apply_transforms(transforms, proteins, tokenizer):
    for transform in transforms or []:
        proteins = transform(proteins, tokenizer=tokenizer)
    return proteins
