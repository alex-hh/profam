from typing import List, Optional

import numpy as np
import torch


def calc_grad_norm(params):
    grad_norm = torch.norm(
        torch.stack(
            [torch.norm(p.grad.detach(), 2) for p in params if p.grad is not None]
        ),
        2,
    )

    return grad_norm


def accuracy_from_outputs(
    model_outputs,
    labels,
    start_ix=0,
    ignore_index=-100,
    dataset_names=None,
    ignore_token_ids: Optional[List[int]] = None,
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
    # Ensure tensors are on the same device
    shift_labels = shift_labels.to(shift_logits.device)
    non_padding_mask = shift_labels != ignore_index
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


def log_likelihood_from_outputs(model_outputs, labels, start_ix=0, flatten=False):
    """Compute the negative log likelihood of the target sequence given the model outputs.

    Args:
        model_outputs: The model outputs from the forward pass.
        input_ids: The input sequence.

    Returns:
        The negative log likelihood of the target sequence.
    """
    logits = model_outputs.logits
    # https://github.com/huggingface/transformers/blob/4a6024921fa142f28e8d0034ae28693713b3bfd0/src/transformers/models/mistral/modeling_mistral.py#L1210

    # Shift so that tokens < n predict n
    shift_logits = logits[..., start_ix:-1, :].contiguous()  # b, L, V
    shift_labels = labels[..., start_ix + 1 :].contiguous()  # b, L
    # Ensure tensors are on the same device
    shift_labels = shift_labels.to(shift_logits.device)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    if flatten:
        # Flatten the tokens
        shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
        shift_labels = shift_labels.view(-1)
        log_likelihood = -loss_fct(shift_logits, shift_labels)
    else:
        log_likelihood = -loss_fct(shift_logits.permute(0, 2, 1), shift_labels)

    return log_likelihood
