import os

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from src.constants import BASEDIR, VOCAB_SIZE


def load_named_model(model_name, overrides=None):
    with initialize_config_dir(os.path.join(BASEDIR, "configs"), version_base="1.3"):
        model_overrides = [f"+constants.vocab_size={VOCAB_SIZE}"] + (overrides or [])
        model_cfg = compose(
            config_name=f"model/{model_name}", overrides=model_overrides
        )
        tokenizer_cfg = compose(config_name=f"tokenizer/profam")

    tokenizer = instantiate(tokenizer_cfg.tokenizer)
    model = instantiate(model_cfg.model, tokenizer=tokenizer)
    return model


def calc_grad_norm(params):
    grad_norm = torch.norm(
        torch.stack(
            [torch.norm(p.grad.detach(), 2) for p in params if p.grad is not None]
        ),
        2,
    )

    return grad_norm


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
