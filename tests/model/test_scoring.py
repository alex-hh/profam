import numpy as np
import torch

from src.models.utils import log_likelihood_from_outputs


def test_kv_cache_no_seqpos(test_model_noseqpos, proteingym_batch):
    model = test_model_noseqpos.eval()
    full_input_ids = torch.cat(
        [proteingym_batch["input_ids"], proteingym_batch["completion_ids"][:, 0]], dim=1
    )
    completion_start_ix = (
        proteingym_batch["input_ids"].shape[1] + 1
    )  # skip the SEP token
    assert full_input_ids[..., completion_start_ix - 1] == model.tokenizer.sep_token_id
    past_key_values = None
    with torch.no_grad():
        outputs = model(full_input_ids, use_cache=False)
        logits_v1 = outputs.logits
        log_likelihood_v1 = log_likelihood_from_outputs(
            outputs, full_input_ids, start_ix=completion_start_ix - 1
        )

    # next run forward pass, caching the kv states
    # input_ids = torch.cat([batch["input_ids"], batch["completion_ids"][:, 0]], dim=1)
    past_key_values = None
    with torch.no_grad():
        outputs = model(
            proteingym_batch["input_ids"],
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

    # # assert len(past_key_values) == config.num_hidden_layers
    # assert len(past_key_values[0]) == 2  # tuple (k, v)
    # assert past_key_values[0][0].shape == (
    #     batch_size,
    #     config.num_key_value_heads,
    #     batch["input_ids"].shape[-1],
    #     config.hidden_size // config.num_attention_heads,
    # )

    # the 0 index is the first mutated sequence
    print(proteingym_batch["completion_ids"][:, 0].shape)
    with torch.no_grad():
        outputs = model.model(
            proteingym_batch["completion_ids"][:, 0],
            past_key_values=past_key_values,
            use_cache=True,
        )
    logits_v2 = outputs.logits
    log_likelihood_v2 = log_likelihood_from_outputs(
        outputs, proteingym_batch["completion_ids"][:, 0]
    )

    assert torch.isclose(log_likelihood_v1, log_likelihood_v2).all()


def test_kv_cache_with_seqpos(test_model, proteingym_batch):
    model = test_model.eval()
    print(proteingym_batch.keys())
    full_input_ids = torch.cat(
        [proteingym_batch["input_ids"], proteingym_batch["completion_ids"][:, 0]], dim=1
    )
    completion_start_ix = (
        proteingym_batch["input_ids"].shape[1] + 1
    )  # skip the SEP token
    assert full_input_ids[..., completion_start_ix - 1] == model.tokenizer.sep_token_id
    past_key_values = None
    with torch.no_grad():
        outputs = model(
            full_input_ids,
            use_cache=False,
        )
        logits_v1 = outputs.logits
        log_likelihood_v1 = log_likelihood_from_outputs(
            outputs, full_input_ids, start_ix=completion_start_ix - 1
        )

    # next run forward pass, caching the kv states
    # input_ids = torch.cat([batch["input_ids"], batch["completion_ids"][:, 0]], dim=1)
    past_key_values = None
    n_seps = int((proteingym_batch["input_ids"] == model.tokenizer.sep_token_id).sum())
    with torch.no_grad():
        outputs = model.model(
            proteingym_batch["input_ids"],
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

    # the 0 index is the first mutated sequence
    print(proteingym_batch["completion_ids"][:, 0].shape)
    with torch.no_grad():
        outputs = model.model(
            proteingym_batch["completion_ids"][:, 0],
            past_key_values=past_key_values,
            use_cache=True,
        )
    logits_v2 = outputs.logits
    log_likelihood_v2 = log_likelihood_from_outputs(
        outputs, proteingym_batch["completion_ids"][:, 0]
    )
    assert torch.isclose(log_likelihood_v1, log_likelihood_v2).all()


def test_seq_scoring(test_model, proteingym_batch):
    model = test_model.eval()
    with torch.no_grad():
        kv_scores = model.score_seqs(
            input_ids=proteingym_batch["input_ids"],
            completion_ids=proteingym_batch["completion_ids"][:, :2],
            use_cache=True,
            batch_size=1,
        )

        scores = model.score_seqs(
            input_ids=proteingym_batch["input_ids"],
            completion_ids=proteingym_batch["completion_ids"][:, :2],
            use_cache=False,
            batch_size=1,
        )
        assert np.isclose(kv_scores, scores).all(), f"{kv_scores} {scores}"


def test_seq_scoring_embed_seq_index(model_seq_index, proteingym_batch):
    model = model_seq_index.eval()
    with torch.no_grad():
        kv_scores = model.score_seqs(
            input_ids=proteingym_batch["input_ids"],
            completion_ids=proteingym_batch["completion_ids"][:, :2],
            use_cache=True,
            batch_size=1,
        )

        scores = model.score_seqs(
            input_ids=proteingym_batch["input_ids"],
            completion_ids=proteingym_batch["completion_ids"][:, :2],
            use_cache=False,
            batch_size=1,
        )
        assert np.isclose(kv_scores, scores).all(), f"{kv_scores} {scores}"


def test_seq_scoring_batched(test_model, proteingym_batch):
    model = test_model.eval()
    with torch.no_grad():
        kv_scores = model.score_seqs(
            input_ids=proteingym_batch["input_ids"],
            completion_ids=proteingym_batch["completion_ids"][:, :4],
            use_cache=True,
            batch_size=2,
        )

        scores = model.score_seqs(
            input_ids=proteingym_batch["input_ids"],
            completion_ids=proteingym_batch["completion_ids"][:, :4],
            use_cache=False,
            batch_size=1,
        )
        assert np.isclose(kv_scores, scores).all(), f"{kv_scores} {scores}"


def test_seq_scoring_noseqpos(test_model_noseqpos, proteingym_batch):
    model = test_model_noseqpos.eval()
    with torch.no_grad():
        kv_scores = model.score_seqs(
            input_ids=proteingym_batch["input_ids"],
            completion_ids=proteingym_batch["completion_ids"][:, :4],
            use_cache=True,
            batch_size=1,
        )

        scores = model.score_seqs(
            input_ids=proteingym_batch["input_ids"],
            completion_ids=proteingym_batch["completion_ids"][:, :4],
            use_cache=False,
            batch_size=1,
        )
        assert np.isclose(kv_scores, scores).all(), f"{kv_scores} {scores}"
