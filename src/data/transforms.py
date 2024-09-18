import bisect
import itertools
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from src.data.fasta import convert_sequence_with_positions
from src.data.objects import InterleavedProteinDocument, Protein, ProteinDocument
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


def fill_missing_fields(
    proteins: ProteinDocument, tokenizer: ProFamTokenizer, **kwargs
):
    # TODO: use DEFAULT FILL VALUES
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
    **kwargs,
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


def concatenate_interleaved_proteins(
    interleaved_proteins: InterleavedProteinDocument, **kwargs
):
    return interleaved_proteins.to_protein_document()


def interleave_proteins(
    proteins: ProteinDocument,
    tokenizer: ProFamTokenizer,
    max_tokens: int,
    **kwargs,
):
    """Interleave proteins, and slice to tokenizer.max tokens."""
    # TODO: rename seq_struct_sep_token -> interleaved_sep_token
    interleaved_proteins = proteins.interleave(
        sequence_separator=tokenizer.seq_struct_sep_token
    )
    total_length = 0
    for i, (prefix, suffix) in enumerate(interleaved_proteins):
        total_length += len(prefix) + len(suffix)
        if total_length > (max_tokens or 1e8):
            break
    return interleaved_proteins[:i]


def mask_interleave_structure_sequence(
    interleaved_proteins: InterleavedProteinDocument,
    tokenizer: ProFamTokenizer,
    structure_first_prob: float = 1.0,
    use_structure_tokens: bool = False,
    **kwargs,
):
    """Fully mask one modality in prefix and other modality in suffix.

    Sequence  | Structure | Task
    ---------------------------
    Masked    | Known     | Inverse folding
    Known     | Masked    | Structure prediction

    This solves inverse folding and structure prediction.
    """
    coin_flip = np.random.rand()
    prefix_proteins = ProteinDocument(interleaved_proteins.prefix_proteins)
    suffix_proteins = ProteinDocument(interleaved_proteins.suffix_proteins)
    if coin_flip < structure_first_prob:
        if use_structure_tokens:
            assert prefix_proteins.structure_tokens is not None
            prefix_proteins = prefix_proteins.clone(
                sequences=prefix_proteins.structure_tokens
            )
        else:
            prefix_proteins = prefix_proteins.mask_sequences(tokenizer.mask_token)
        # masks and plddts retained to identify structure information that is available in prefix
        suffix_proteins = suffix_proteins.mask_structures(
            mask_token=tokenizer.mask_token, plddts=False, backbone_coords_masks=False
        )
    else:
        # suffix has no information at all about structure so mask masks and plddts to
        prefix_proteins = prefix_proteins.mask_structures(
            mask_token=tokenizer.mask_token, plddts=True, backbone_coords_masks=True
        )
        if use_structure_tokens:
            assert suffix_proteins.structure_tokens is not None
            suffix_proteins = suffix_proteins.clone(
                sequences=suffix_proteins.structure_tokens
            )
        else:
            suffix_proteins = suffix_proteins.mask_sequences(tokenizer.mask_token)
    return interleaved_proteins.clone(prefix_proteins, suffix_proteins)


def partially_mask_prefixes_for_protein_design(
    interleaved_proteins: InterleavedProteinDocument,
    tokenizer: ProFamTokenizer,
    structure_mask_prob_bounds: Tuple = (0.0, 1.0),
    sequence_sub_mask_prob_bounds: Tuple = (0.0, 1.0),
    remove_unmasked_sequence_positions_from_suffix: bool = False,
    **kwargs,
):
    """Randomly mask a subset of positions in prefix.

    TODO: in model, we need to use suffix mask to prevent computing loss on prefix aas.

    Design-general masking: mask a given fraction of coordinates,
    mask the sequence at a subset of those positions.

    Entirely mask the structure in the suffix.

    | Sequence| Structure | Task
    ---------------------------
    Masked    | Known     | Inverse folding
    Mask A    | Mask A    | Design (Motif scaffolding / De novo design)
    Mask subA | Mask A    | Partially constrained redesign (inverse folding + de novo)
    """
    prefix_proteins = ProteinDocument(interleaved_proteins.prefix_proteins)
    # mask out structure in suffix
    suffix_proteins = ProteinDocument(
        interleaved_proteins.suffix_proteins
    ).mask_structure(plddts=False, backbone_coords_masks=False)
    prefix_structure_masks = []
    for protein in prefix_proteins:
        structure_mask_prob = np.random.uniform(*structure_mask_prob_bounds)
        structure_mask = np.random.rand(len(protein)) < structure_mask_prob
        prefix_structure_masks.append(structure_mask)

    prefix_proteins = prefix_proteins.mask_structures(
        mask_token=tokenizer.mask_token,
        plddts=True,
        backbone_structure_masks=True,
        masks=prefix_structure_masks,
    )
    # now build sequence masks. we mask all positions that are already masked, and a subset of remaining positions
    prefix_sequence_masks = []
    for structure_mask in prefix_structure_masks:
        sequence_sub_mask_prob = np.random.uniform(*sequence_sub_mask_prob_bounds)
        sequence_mask = (
            np.where(structure_mask, 0.0, np.random.rand(len(structure_mask)))
            < sequence_sub_mask_prob
        )
        prefix_sequence_masks.append(sequence_mask)

    # apply the same mask to the suffix coords masks so that they are aware of the prefix mask
    # i.e. suffix coords mask should be True if the corresponding position in the prefix is masked
    suffix_proteins = suffix_proteins.mask_structures(
        mask_token=tokenizer.mask_token,
        plddts=True,
        backbone_structure_masks=True,
        masks=prefix_structure_masks,
    )

    prefix_proteins = prefix_proteins.mask_sequences(
        mask_token=tokenizer.mask_token, masks=prefix_sequence_masks
    )
    if remove_unmasked_sequence_positions_from_suffix:
        suffix_proteins = suffix_proteins.masked_slice_proteins(
            [~m for m in prefix_sequence_masks]
        )

    return interleaved_proteins.clone(prefix_proteins, suffix_proteins)


def partially_mask_prefixes_for_structure_prediction(
    interleaved_proteins: InterleavedProteinDocument,
    tokenizer: ProFamTokenizer,
    sequence_mask_prob_bounds: Tuple = (0.5, 1.0),
    structure_sub_mask_prob_bounds: Tuple = (0.0, 0.0),
    remove_unmasked_sequence_positions_from_suffix: bool = True,
    **kwargs,
):
    """Converse of `partially_mask_prefixes_for_sequence_design`.

    N.B. we still have a sequence only-suffix, since we assume for now
    that structure prediction loss is applied to the prefix.

    Since we usually want to predict entire structure, we mask entire structure by default.
    """
    prefix_proteins = ProteinDocument(interleaved_proteins.prefix_proteins)
    # mask out structure in suffix
    suffix_proteins = ProteinDocument(
        interleaved_proteins.suffix_proteins
    ).mask_structure(plddts=False, backbone_coords_masks=False)
    prefix_sequence_masks = []
    for protein in prefix_proteins:
        sequence_mask_prob = np.random.uniform(*sequence_mask_prob_bounds)
        sequence_mask = np.random.rand(len(protein)) < sequence_mask_prob
        prefix_sequence_masks.append(sequence_mask)

    prefix_proteins = prefix_proteins.mask_sequences(
        tokenizer.mask_token, masks=prefix_sequence_masks
    )
    # now build sequence masks. we mask all positions that are already masked, and a subset of remaining positions
    prefix_structure_masks = []
    for sequence_mask in prefix_sequence_masks:
        structure_sub_mask_prob = np.random.uniform(*structure_sub_mask_prob_bounds)
        structure_mask = (
            np.where(sequence_mask, 0.0, np.random.rand(len(sequence_mask)))
            < structure_sub_mask_prob
        )
        prefix_structure_masks.append(structure_mask)

    prefix_proteins = prefix_proteins.mask_structures(
        mask_token=tokenizer.mask_token,
        plddts=True,
        backbone_coords_masks=True,
        masks=prefix_structure_masks,
    )
    suffix_proteins = suffix_proteins.mask_structures(
        mask_token=tokenizer.mask_token,
        plddts=True,
        backbone_coords_masks=True,
        masks=prefix_structure_masks,
    )
    if remove_unmasked_sequence_positions_from_suffix:
        suffix_proteins = suffix_proteins.masked_slice_proteins(
            [~m for m in prefix_sequence_masks]
        )

    return interleaved_proteins.clone(prefix_proteins, suffix_proteins)


def replace_selenocysteine_pyrrolysine(proteins: ProteinDocument, **kwargs):
    new_sequences = [
        seq.replace("U", "C").replace("O", "K") for seq in proteins.sequences
    ]
    return proteins.clone(sequences=new_sequences)


def apply_transforms(transforms, proteins, tokenizer, **kwargs):
    # TODO: consider passing max tokens as kwarg, instead of inferring from tokenizer,
    # where it doesn't really belong
    for transform in transforms or []:
        proteins = transform(proteins, tokenizer=tokenizer, **kwargs)
    return proteins
