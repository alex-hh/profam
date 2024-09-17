from typing import List, Optional

import numpy as np
import torch


def has_coords_frac(coords_mask, structure_mask, **kwargs):
    has_coords_mask = (
        coords_mask.any((-1, -2)) & structure_mask
    )  # and structure mask probably not necessary
    assert has_coords_mask.ndim == 2  # b, L
    has_coords_frac = has_coords_mask.float().sum() / structure_mask.float().sum()
    return has_coords_frac


def plddt_metrics(
    plddts, structure_mask: torch.Tensor, coords_mask: torch.Tensor, **kwargs
):
    metrics = {}
    has_coords_mask = coords_mask.any((-1, -2)) & structure_mask
    mean_plddt_unmasked = (
        plddts * has_coords_mask.float()
    ).sum() / has_coords_mask.float().sum()
    metrics["mean_plddt_unmasked"] = mean_plddt_unmasked
    metrics["mean_plddt"] = (
        plddts * structure_mask.float()
    ).sum() / structure_mask.float().sum()
    return metrics


def accuracy_from_outputs(
    model_outputs,
    labels,
    start_ix=0,
    ignore_index=-100,
    dataset_names=None,
    ignore_token_ids: Optional[List[int]] = None,
    mask=None,
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
    if dataset_names is not None:
        # N.B. this also works for empty list
        ds_accuracies = {}
        for ds_name in set(dataset_names):
            in_dataset_mask = np.array(dataset_names) == ds_name
            ds_accuracies[ds_name] = (
                accuracy[in_dataset_mask] * non_padding_mask[in_dataset_mask]
            ).sum() / non_padding_mask[in_dataset_mask].sum()
        ds_accuracies["global"] = (
            accuracy * non_padding_mask
        ).sum() / non_padding_mask.sum()
        return ds_accuracies
    accuracy = (accuracy * non_padding_mask).sum() / non_padding_mask.sum()
    return accuracy
