import bisect
import itertools
from typing import Optional

import numpy as np

from src.data.fasta import convert_sequence_with_positions
from src.data.objects import ProteinDocument
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
    # TODO: implement keep first, drop first
    if drop_first:
        proteins = proteins[1:]

    if shuffle:
        perm = rnd.permutation(len(proteins))
        proteins = proteins[perm]

    if max_tokens is not None:
        cumulative_lengths = list(
            itertools.accumulate(
                [len(s) + extra_tokens_per_sequence for s in proteins.sequences]
            )
        )  # +1 for separator
        insertion_point = bisect.bisect_left(
            cumulative_lengths,
            max_tokens - extra_tokens_per_document,
        )
    else:
        insertion_point = len(proteins)
    proteins = proteins[:insertion_point]
    return proteins


def fill_missing_fields(proteins: ProteinDocument):
    if not proteins.has_all_structure_arrays:
        proteins = proteins.fill_missing_structure_arrays(
            coords_fill=np.nan,
            plddts_fill=100.0,
            tokens_fill="[MASK]",
        )
    return proteins


def interleave_structure_sequence(
    proteins: ProteinDocument,
    tokenizer: ProFamTokenizer,
    max_tokens: int,
    structure_first_prob: float = 1.0,
):
    assert proteins.has_all_structure_arrays
    coin_flip = np.random.rand()
    interleaved_sequences = []
    interleaved_positions = []
    interleaved_plddts = []
    interleaved_coords = []
    interleaved_coords_masks = []
    total_tokens = tokenizer.num_start_tokens
    for seq, seq_3d, xyz, coords_mask, plddts, positions in zip(
        proteins.sequences,
        proteins.structure_tokens,
        proteins.backbone_coords,
        proteins.backbone_coords_masks,
        proteins.plddts,
        proteins.positions,
    ):
        # TODO: monitor max_tokens
        assert len(seq) == len(seq_3d) == len(xyz) == len(plddts)
        assert isinstance(positions, list)
        if coin_flip < structure_first_prob:
            interleaved_sequences.append(seq_3d + tokenizer.seq_struct_sep_token + seq)
            interleaved_positions.append(positions + [0] + positions)
            interleaved_plddts.append(
                np.concatenate(
                    [np.array(plddts), np.full((1,), 100.0), np.array(plddts)]
                )
            )
            interleaved_coords.append(
                np.concatenate([xyz, np.full((1, 4, 3), np.nan), xyz], axis=0)
            )
            interleaved_coords_masks.append(
                np.concatenate(
                    [coords_mask, np.zeros((1, 4, 3)), np.zeros_like(xyz)], axis=0
                )
            )
        else:
            interleaved_sequences.append(seq + tokenizer.seq_struct_sep_token + seq_3d)
            interleaved_positions.append(positions + [0] + positions)
            interleaved_plddts.append(
                np.concatenate(
                    [np.array(plddts), np.full((1,), 100.0), np.array(plddts)]
                )
            )
            interleaved_coords.append(
                np.concatenate([xyz, np.full((1, 4, 3), np.nan), xyz], axis=0)
            )
            interleaved_coords_masks.append(
                np.concatenate(
                    [np.zeros_like(xyz), np.zeros((1, 4, 3)), coords_mask], axis=0
                )
            )

        total_tokens += len(seq) + len(seq_3d) + 2  # +1 for each separator

        if total_tokens > max_tokens:
            interleaved_sequences = interleaved_sequences[:-1]
            interleaved_positions = interleaved_positions[:-1]
            interleaved_plddts = interleaved_plddts[:-1]
            interleaved_coords = interleaved_coords[:-1]
            interleaved_coords_masks = interleaved_coords_masks[:-1]
            assert (
                len(interleaved_sequences) > 0
            ), "Cannot fit any sequences in max_tokens"
            break

    return proteins.clone(
        accessions=[f"seq{i}" for i in range(len(interleaved_sequences))],
        sequences=interleaved_sequences,
        positions=interleaved_positions,
        plddts=interleaved_plddts,
        backbone_coords=interleaved_coords,
        backbone_coords_masks=interleaved_coords_masks,
        structure_tokens=None,
        validate_shapes=False,  # a hack because of special token in interleaved sequences
    )


def apply_transforms(transforms, proteins, tokenizer):
    for transform in transforms or []:
        proteins = transform(proteins, tokenizer=tokenizer)
    return proteins
