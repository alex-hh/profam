import pytest
import torch

from src.models import metrics


def test_single_doc_single_batch():
    """Test sequence packing case (batch_size=1 with single document multiple sequences)"""
    # Document structure: [BOS] seq1 [SEP] seq2 [SEP] seq3 [SEP]
    labels = torch.tensor([[1, 2, 3, 4, 5, 3, 4, 5, 3, 4]])

    # Make perfect predictions
    logits = torch.zeros(1, labels.shape[-1], 100)  # (batch, seq_len, vocab)
    logits[0, :, :] = -100
    for seq_ix, label in enumerate(labels[0, 1:]):
        if label == -100:
            continue
        logits[0, seq_ix, label] = 100

    outputs = type("", (), {"logits": logits})()
    acc_metrics = metrics.accuracy_from_outputs(
        input_ids=labels,
        model_outputs=outputs,
        labels=labels,
        start_ix=0,
        sep_token_id=4,
        bos_token_id=1,
        calc_full_no_context_accuracies=True,
    )
    assert torch.isclose(acc_metrics["global"], torch.tensor(1.0))
    assert torch.isclose(
        acc_metrics["first_sequence"], torch.tensor(1.0)
    )  # positions 1-3
    assert torch.isclose(
        acc_metrics["last_sequence"], torch.tensor(1.0)
    )  # positions 7-9

    logits[0, -3, 3] = -1000  # incorrect pred
    acc_metrics = metrics.accuracy_from_outputs(
        input_ids=labels,
        model_outputs=outputs,
        labels=labels,
        start_ix=0,
        sep_token_id=4,
        bos_token_id=1,
        calc_full_no_context_accuracies=True,
    )
    assert torch.isclose(acc_metrics["last_sequence"], torch.tensor(2 / 3))

    logits[0, 2, 4] = -1000  # incorrect at first sep
    logits[0, 1, 3] = -1000  # incorrect
    acc_metrics = metrics.accuracy_from_outputs(
        input_ids=labels,
        model_outputs=outputs,
        labels=labels,
        start_ix=0,
        sep_token_id=4,
        bos_token_id=1,
        calc_full_no_context_accuracies=True,
    )
    assert torch.isclose(acc_metrics["first_sequence"], torch.tensor(1 / 3))
    assert torch.isclose(acc_metrics["last_sequence"], torch.tensor(2 / 3))
    assert torch.isclose(acc_metrics["global"], torch.tensor((1 + 3 + 2) / 9))


def test_sequence_packing():
    """Test sequence packing case (batch_size=1 with single document multiple sequences)"""
    # Document structure: [BOS] seq1 [SEP] seq2 [SEP] seq3 [SEP]
    # fmt: off
    labels = torch.tensor([[1, 2, 2, 4, 2, 2, 4,  #2seq doc
                            1, 2, 2, 4, 2, 2, 2, 4, #2seq doc
                            1, 2, 4, 2, 2, 4, 2, 2, 2, 4 #3seq doc
                            ]])
    # fmt: on
    total_predictable_tokens = labels.shape[-1] - 3
    # Make perfect predictions
    logits = torch.zeros(1, labels.shape[-1], 100)  # (batch, seq_len, vocab)
    logits[0, :, :] = -100
    for seq_ix, label in enumerate(labels[0, 1:]):
        if label == -100:
            continue
        logits[0, seq_ix, label] = 100

    outputs = type("", (), {"logits": logits})()
    acc_metrics = metrics.accuracy_from_outputs(
        input_ids=labels,
        model_outputs=outputs,
        labels=labels,
        start_ix=0,
        sep_token_id=4,
        bos_token_id=1,
        calc_full_no_context_accuracies=True,
    )
    assert torch.isclose(acc_metrics["global"], torch.tensor(1.0))
    assert torch.isclose(acc_metrics["first_sequence"], torch.tensor(1.0))
    assert torch.isclose(acc_metrics["last_sequence"], torch.tensor(1.0))

    logits[0, 0, 2] = -1000  # incorrect pred first seq 1
    acc_metrics = metrics.accuracy_from_outputs(
        input_ids=labels,
        model_outputs=outputs,
        labels=labels,
        start_ix=0,
        sep_token_id=4,
        bos_token_id=1,
        calc_full_no_context_accuracies=True,
    )
    assert torch.isclose(acc_metrics["last_sequence"], torch.tensor(1.0))
    assert torch.isclose(
        acc_metrics["first_sequence"], torch.tensor((2 + 3 + 2) / (3 + 3 + 2))
    )
    global_acc = acc_metrics["global"]

    # check that bos_token_id is ignored: no change in accuracy:
    logits[0, 6, 99] = 1000
    logits[0, 14, 99] = 1000
    assert torch.isclose(acc_metrics["last_sequence"], torch.tensor(1.0))
    assert torch.isclose(
        acc_metrics["first_sequence"], torch.tensor((2 + 3 + 2) / (3 + 3 + 2))
    )
    assert torch.isclose(acc_metrics["global"], global_acc)

    logits[0, 1, 99] = 1000
    logits[0, 8, 99] = 1000
    logits[0, -2, 99] = 1000
    acc_metrics = metrics.accuracy_from_outputs(
        input_ids=labels,
        model_outputs=outputs,
        labels=labels,
        start_ix=0,
        sep_token_id=4,
        bos_token_id=1,
        calc_full_no_context_accuracies=True,
    )
    assert torch.isclose(
        acc_metrics["first_sequence"], torch.tensor((1 + 2 + 2) / (3 + 3 + 2))
    )
    assert torch.isclose(
        acc_metrics["last_sequence"], torch.tensor((3 + 4 + 3)) / (3 + 4 + 4)
    )
    assert torch.isclose(
        acc_metrics["global"],
        torch.tensor((total_predictable_tokens - 4) / total_predictable_tokens),
    )


def test_multiple_documents_batch_dim():
    """Test batch_size >1 with separate documents in batch dimension"""
    # Batch 0: [BOS] A [SEP] B [SEP]
    # Batch 1: [BOS] C [SEP]
    # fmt: off
    labels_list = [
        torch.tensor([
        [1, 2, 3, 4, 2,    3,    4   ],
        [1, 5, 3, 4, -100, -100, -100]
        ]),

        torch.tensor([
            [1, 2, 3, 5, 6, 4, 6, 5, 4, 3, 4, 3, 2, 4],
            [1, 2, 2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 4, -100]
        ]),
    ]
    # fmt: on
    for labels in labels_list:
        # Create logits with perfect predictions
        logits = torch.zeros(labels.shape[0], labels.shape[1], 100)
        for b in range(labels.shape[0]):
            for seq_ix, label in enumerate(labels[b, 1:]):
                if label == -100:
                    continue
                logits[b, seq_ix, label] = 100

        outputs = type("", (), {"logits": logits})()
        acc_metrics = metrics.accuracy_from_outputs(
            input_ids=labels,
            model_outputs=outputs,
            labels=labels,
            start_ix=0,
            sep_token_id=4,
            bos_token_id=1,
            dataset_names=["ds1", "ds2"],
            calc_full_no_context_accuracies=True,
        )

        # Global accuracy should be (5+2 correct)/(5+2 total)
        assert torch.isclose(acc_metrics["global"], torch.tensor(1.0))
        # First sequence accuracies
        assert torch.isclose(
            acc_metrics["ds1_first_sequence"], torch.tensor(1.0)
        )  # positions 1-2
        assert torch.isclose(
            acc_metrics["ds2_first_sequence"], torch.tensor(1.0)
        )  # position 1

    # Now introduce some errors
    logits[0, 1, 3] = -1000  # incorrect in first batch, first sequence
    logits[1, 2, 2] = -1000  # incorrect in second batch, first sequence

    acc_metrics = metrics.accuracy_from_outputs(
        input_ids=labels,
        model_outputs=outputs,
        labels=labels,
        start_ix=0,
        sep_token_id=4,
        bos_token_id=1,
        dataset_names=["ds1", "ds2"],
        calc_full_no_context_accuracies=True,
    )

    # First sequence accuracies with errors
    assert torch.isclose(acc_metrics["ds1_first_sequence"], torch.tensor(0.8))
    assert torch.isclose(acc_metrics["ds2_first_sequence"], torch.tensor(0.75))
    assert torch.isclose(acc_metrics["ds1_last_sequence"], torch.tensor(1.0))
    assert torch.isclose(acc_metrics["ds2_last_sequence"], torch.tensor(1.0))
    # Global accuracy: (total correct)/(total predictable)

    total_predictable = labels.shape[-1] * 2 - 3
    total_correct = total_predictable - 2  # 2 errors
    assert torch.isclose(
        acc_metrics["global"], torch.tensor(total_correct / total_predictable)
    )


def test_single_sequence_document():
    """Test documents with only one sequence (no SEP tokens)"""
    # fmt: off
    labels = torch.tensor([
        [1, 2, 3, 4, -100, -100],  # Single sequence
        [1, 4, 5, 6, 4,    -100] # first seq is just a SEP
    ])
    # fmt: on

    logits = torch.zeros(labels.shape[0], labels.shape[1], 100)
    for b in range(labels.shape[0]):
        for seq_ix, label in enumerate(labels[b, 1:]):
            if label == -100:
                continue
            logits[b, seq_ix, label] = 100

    outputs = type("", (), {"logits": logits})()
    acc_metrics = metrics.accuracy_from_outputs(
        input_ids=labels,
        model_outputs=outputs,
        labels=labels,
        start_ix=0,
        sep_token_id=4,
        bos_token_id=1,
        calc_full_no_context_accuracies=True,
    )

    # First and last sequence should match global
    assert acc_metrics["global"] == acc_metrics["first_sequence"]
    assert acc_metrics["first_sequence"] == acc_metrics["last_sequence"]

    # Introduce errors
    logits[0, 1, 3] = -1000  # incorrect in first sequence
    logits[1, 2, 6] = -1000  # incorrect in second sequence

    acc_metrics = metrics.accuracy_from_outputs(
        input_ids=labels,
        model_outputs=outputs,
        labels=labels,
        start_ix=0,
        sep_token_id=4,
        bos_token_id=1,
        calc_full_no_context_accuracies=True,
    )

    assert torch.isclose(acc_metrics["global"], torch.tensor(5 / 7))
    assert torch.isclose(acc_metrics["first_sequence"], torch.tensor((2 + 1) / (3 + 1)))
    assert torch.isclose(acc_metrics["last_sequence"], torch.tensor((2 + 2) / (3 + 3)))


def test_padding_handling():
    """Test proper ignoring of padding tokens"""
    # fmt: off
    labels = torch.tensor([
        [1, 2, 3, 4,    -100, -100],
        [1, 5, 4, -100, -100, -100]
    ])
    # fmt: on

    # Create logits with some errors
    logits = torch.zeros(labels.shape[0], labels.shape[-1], 100)
    logits[0, 0, 2] = 100  # correct
    logits[0, 1, 3] = 100  # correct
    logits[0, 2, 0] = 100  # incorrect
    logits[1, 0, 5] = 100  # correct
    logits[1, 1, 4] = 100  # correct

    outputs = type("", (), {"logits": logits})()
    acc_metrics = metrics.accuracy_from_outputs(
        input_ids=labels,
        model_outputs=outputs,
        labels=labels,
        start_ix=0,
        sep_token_id=4,
        bos_token_id=1,
    )

    assert torch.isclose(acc_metrics["global"], torch.tensor(0.8))

    logits[0, 0, 2] = -100  # incorrect
    acc_metrics = metrics.accuracy_from_outputs(
        input_ids=labels,
        model_outputs=outputs,
        labels=labels,
        start_ix=0,
        sep_token_id=4,
        bos_token_id=1,
    )
    # Expected accuracy: (3 correct, 2 incorrect)
    assert torch.isclose(acc_metrics["global"], torch.tensor(0.6))

    logits[0, 1, 3] = -100  # incorrect
    acc_metrics = metrics.accuracy_from_outputs(
        input_ids=labels,
        model_outputs=outputs,
        labels=labels,
        start_ix=0,
        sep_token_id=4,
        bos_token_id=1,
    )
    # Expected accuracy: (2 correct, 3 incorrect)
    assert torch.isclose(acc_metrics["global"], torch.tensor(0.4))

    logits[1, 0, 5] = -100  # incorrect
    acc_metrics = metrics.accuracy_from_outputs(
        input_ids=labels,
        model_outputs=outputs,
        labels=labels,
        start_ix=0,
        sep_token_id=4,
        bos_token_id=1,
    )
    # Expected accuracy: (1 correct, 4 incorrect)
    assert torch.isclose(acc_metrics["global"], torch.tensor(0.2))


def test_mixed_dataset_batch():
    """Test batch with mixed dataset examples"""
    # fmt: off
    labels = torch.tensor([
        [1, 2, 3, 4, 2,    3,    4,    -100],  # ds1
        [1, 5, 3, 4, -100, -100, -100, -100]  # ds2
    ])
    # fmt: on

    logits = torch.zeros(labels.shape[0], labels.shape[-1], 100)
    for b in range(labels.shape[0]):
        for seq_ix, label in enumerate(labels[b, 1:]):
            if label == -100:
                continue
            logits[b, seq_ix, label] = 100

    outputs = type("", (), {"logits": logits})()
    acc_metrics = metrics.accuracy_from_outputs(
        input_ids=labels,
        model_outputs=outputs,
        labels=labels,
        start_ix=0,
        sep_token_id=4,
        bos_token_id=1,
        dataset_names=["ds1", "ds2"],
        calc_full_no_context_accuracies=True,
    )

    # Verify dataset-specific metrics
    assert torch.isclose(acc_metrics["ds1"], torch.tensor(1.0))
    assert torch.isclose(acc_metrics["ds2"], torch.tensor(1.0))
    assert torch.isclose(acc_metrics["ds1_last_sequence"], torch.tensor(1.0))
    assert torch.isclose(acc_metrics["ds2_last_sequence"], torch.tensor(1.0))


def test_edge_cases():
    """Test empty sequences and all-padding cases"""
    # All padding
    labels = torch.full((2, 5), -100)
    logits = torch.randn(labels.shape[0], labels.shape[-1], 100)

    outputs = type("", (), {"logits": logits})()
    acc_metrics = metrics.accuracy_from_outputs(
        input_ids=labels,
        model_outputs=outputs,
        labels=labels,
        start_ix=0,
    )

    # Should return NaN for global accuracy
    assert torch.isnan(acc_metrics["global"])
