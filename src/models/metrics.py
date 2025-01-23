from typing import List, Optional

import numpy as np
import torch


def has_coords_frac(coords_mask, structure_mask, **kwargs):
    assert coords_mask.ndim == 4
    has_coords_mask = (
        coords_mask.flatten(start_dim=-2).any(-1) & structure_mask
    )  # and structure mask probably not necessary
    assert has_coords_mask.ndim == 2  # b, L
    has_coords_frac = has_coords_mask.float().sum() / structure_mask.float().sum()
    return has_coords_frac


def plddt_metrics(
    plddts, structure_mask: torch.Tensor, coords_mask: torch.Tensor, **kwargs
):
    metrics = {}
    assert coords_mask.ndim == 4
    has_coords_mask = coords_mask.flatten(start_dim=-2).any(-1) & structure_mask
    mean_plddt_unmasked = (
        plddts * has_coords_mask.float()
    ).sum() / has_coords_mask.float().sum()
    metrics["mean_plddt_unmasked"] = mean_plddt_unmasked
    metrics["mean_plddt"] = (
        plddts * structure_mask.float()
    ).sum() / structure_mask.float().sum()
    return metrics


def calc_accuracy_with_masks(
    token_accuracy,
    sample_mask: Optional[torch.Tensor] = None,
    token_mask: Optional[torch.Tensor] = None,
):
    if sample_mask is not None:
        token_accuracy = token_accuracy[sample_mask]
        if token_mask is not None:
            token_mask = token_mask[sample_mask]
    if token_mask is not None:
        token_accuracy = token_accuracy * token_mask
    return token_accuracy.sum() / token_mask.sum()


def accuracy_from_outputs(
    model_outputs,
    labels,
    start_ix=0,
    ignore_index=-100,
    dataset_names=None,
    ignore_token_ids: Optional[List[int]] = None,
    mask=None,
    sep_token_id=None,
    calc_full_no_context_accuracies: bool = False,
):
    """Compute the accuracy of the target sequence given the model outputs.

    Args:
        model_outputs: The model outputs from the forward pass.
        input_ids: The input sequence.
        ignore_index: Token index to ignore when computing accuracy.
            (this will get added automatically by the data collator as padding)

    Returns:
        The accuracy of the target sequence.
    """
    if calc_full_no_context_accuracies:
        assert sep_token_id is not None
        # cat ensures that sep token is included in the prev sequence for index
        sequence_indices = torch.cat(
            [
                torch.zeros(labels.shape[0], 1, device=labels.device),
                torch.cumsum(labels == sep_token_id, dim=-1)[:, :-1],
            ],
            dim=-1,
        )
        # TODO: assert that last non-padding token is a sep token (in pre-sliced labels)
        # TODO: write test
        first_sequence_mask = sequence_indices == 0
        last_sequence_mask = (
            sequence_indices == sequence_indices[:, -1:] - 1
        )  # -1 because padding tokens will get extra seq index having seen final sep

        first_sequence_mask = first_sequence_mask[:, start_ix + 1 :]
        last_sequence_mask = last_sequence_mask[:, start_ix + 1 :]

    labels = labels.clone()

    if ignore_token_ids is not None:
        ignore_token_ids = torch.tensor(ignore_token_ids).to(labels.device)
        labels[torch.isin(labels, ignore_token_ids)] = ignore_index
    logits = model_outputs.logits
    # Shift so that tokens < n predict n
    shift_logits = logits[..., start_ix:-1, :].contiguous()  # b, L, V
    shift_labels = labels[..., start_ix + 1 :].contiguous()  # b, L
    if mask is not None:
        # Shift mask to match the shifted labels
        shift_mask = mask[..., start_ix + 1 :]
    # Ensure tensors are on the same device
    shift_labels = shift_labels.to(shift_logits.device)
    non_padding_mask = shift_labels != ignore_index
    if mask is not None:
        non_padding_mask = non_padding_mask & shift_mask
    # TODO: we might also want to ignore gaps...
    accuracy = (shift_logits.argmax(-1) == shift_labels).float()
    global_accuracy = calc_accuracy_with_masks(accuracy, token_mask=non_padding_mask)

    accuracy_metrics = {
        "global": global_accuracy,
    }
    if calc_full_no_context_accuracies:
        accuracy_metrics["first_sequence"] = calc_accuracy_with_masks(
            accuracy,
            token_mask=first_sequence_mask & non_padding_mask,
        )
        accuracy_metrics["last_sequence"] = calc_accuracy_with_masks(
            accuracy,
            token_mask=last_sequence_mask & non_padding_mask,
        )

    if dataset_names is not None:
        # N.B. this also works for empty list
        ds_accuracies = {}
        for ds_name in set(dataset_names):
            in_dataset_mask = np.array(dataset_names) == ds_name
            ds_accuracies[ds_name] = calc_accuracy_with_masks(
                accuracy, sample_mask=in_dataset_mask, token_mask=non_padding_mask
            )
            if calc_full_no_context_accuracies:
                ds_accuracies[ds_name + "_first_sequence"] = calc_accuracy_with_masks(
                    accuracy,
                    sample_mask=in_dataset_mask,
                    token_mask=first_sequence_mask & non_padding_mask,
                )
                ds_accuracies[ds_name + "_last_sequence"] = calc_accuracy_with_masks(
                    accuracy,
                    sample_mask=in_dataset_mask,
                    token_mask=last_sequence_mask & non_padding_mask,
                )

        accuracy_metrics.update(ds_accuracies)

    return accuracy_metrics


def sequence_lengths(labels, sep_token_id):
    sep_mask = labels == sep_token_id
    positions = torch.where(sep_mask)[1]
    sequence_lengths = torch.cat([positions[0], positions.diff(dim=-1)])
    result = {
        "min_seq_length": sequence_lengths.min(),
        "max_seq_length": sequence_lengths.max(),
        "mean_seq_length": sequence_lengths.mean(),
    }
    return result


def document_lengths(labels, start_of_doc_token_id):
    start_of_doc_mask = labels == start_of_doc_token_id
    positions = torch.where(start_of_doc_mask)[1]
    doc_lengths = torch.cat([positions[0], positions.diff(dim=-1)])
    result = {
        "min_doc_length": doc_lengths.min(),
        "max_doc_length": doc_lengths.max(),
        "mean_doc_length": doc_lengths.mean(),
    }
    return result
