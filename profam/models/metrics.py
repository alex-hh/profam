import os
import socket
from typing import List, Optional

import numpy as np
import torch

try:
    # attempt to collect NVIDIA GPU information
    import pynvml

    # # Init
    # pynvml.nvmlInit()
    # num_gpus = pynvml.nvmlDeviceGetCount()
    # # Identify local rank and hostname
    # # local_rank = get_local_rank()
    # hostname = socket.gethostname()
    # print(
    #     f"Logging GPU metrics to wandb: num gpus={num_gpus}, local rank={local_rank}, hostname={hostname}"
    # )
except Exception as e:
    pynvml = None
    print(f"pynvml not installed, GPU metrics will not be logged. Error: {e}")


def has_coords_frac(coords_mask, structure_mask, **kwargs):
    assert coords_mask.ndim == 4
    has_coords_mask = (
        coords_mask.flatten(start_dim=-2).any(-1) & structure_mask
    )  # and structure mask probably not necessary
    assert has_coords_mask.ndim == 2  # b, L
    has_coords_frac = has_coords_mask.float().sum() / structure_mask.float().sum()
    return has_coords_frac


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
    input_ids,
    model_outputs,
    labels,
    start_ix=0,
    ignore_index=-100,
    dataset_names=None,
    ignore_token_ids: Optional[List[int]] = None,
    mask=None,
    sep_token_id=None,
    bos_token_id=None,
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
        assert bos_token_id is not None

        # Calculate document indices using BOS tokens
        document_indices = torch.cumsum(
            input_ids == bos_token_id, dim=-1
        )  # (batch, seq_len)
        # Calculate sequence indices that reset at each document
        sep_mask = (labels == sep_token_id).long()
        sequence_indices = torch.zeros_like(document_indices)
        last_sequence_mask = torch.zeros_like(sequence_indices, dtype=torch.bool)

        for b in range(labels.size(0)):
            unique_docs = torch.unique(document_indices[b])
            for doc in unique_docs:
                doc_mask = document_indices[b] == doc
                doc_span = torch.where(doc_mask)[0]
                if len(doc_span) == 0:
                    continue
                start, end = doc_span[0], doc_span[-1] + 1
                doc_sep = sep_mask[b, start:end]
                one_doc_seq_indices = torch.cat(
                    [
                        torch.tensor([0], device=labels.device),
                        doc_sep.cumsum(dim=0)[:-1],
                    ]
                )
                sequence_indices[b, start:end] = one_doc_seq_indices
                max_seq = one_doc_seq_indices.max()
                if labels[b, doc_mask][-1] == ignore_index:  # last token is padding
                    max_seq = max_seq - 1  # avoid counting padding as seq
                # bitwise OR to ensure we don't overwrite previous max_seqs:
                last_sequence_mask[b] |= (sequence_indices[b] == max_seq) & doc_mask

        first_sequence_mask = sequence_indices == 0
        first_sequence_mask = first_sequence_mask[:, start_ix + 1 :]
        last_sequence_mask = last_sequence_mask[:, start_ix + 1 :]

    labels = labels.clone()

    # Combine bos_token_id with ignore_token_ids
    combined_ignore = []
    if ignore_token_ids is not None:
        combined_ignore.extend(ignore_token_ids)
    if bos_token_id is not None:
        combined_ignore.append(bos_token_id)

    if combined_ignore:
        combined_ignore_tensor = torch.tensor(combined_ignore).to(labels.device)
        labels[torch.isin(labels, combined_ignore_tensor)] = ignore_index

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
    sequence_lengths = torch.cat([positions[0].unsqueeze(0), positions.diff(dim=-1)])
    result = {
        "min_seq_length": sequence_lengths.min().item(),
        "max_seq_length": sequence_lengths.max().item(),
        "mean_seq_length": sequence_lengths.float().mean().item(),
    }
    return result
