from collections import defaultdict

import numpy as np


def uniformly_sample_clusters(
    sequences, cluster_ids, max_total_length, tokens_per_sequence=1
):
    # Step 1: Group sequences by their attributes
    clusters = defaultdict(list)
    for ix, (seq, cl_id) in enumerate(zip(sequences, cluster_ids)):
        clusters[cl_id].append((ix, seq))

    selected_ids = []
    total_length = 0

    # Step 2: Sample the same number of items from each stratum
    while True:
        unique_cluster_ids = list(clusters.keys())
        cluster = np.random.choice(unique_cluster_ids)
        candidates = clusters[cluster]
        ix, seq = candidates.pop(np.random.choice(len(candidates)))

        if (
            total_length + len(seq) + tokens_per_sequence > max_total_length
            or not clusters
        ):
            break

        selected_ids.append(ix)
        total_length += len(seq) + tokens_per_sequence

        if not candidates:
            del clusters[cluster]

    return selected_ids


def filter_on_length(
    example,
    max_tokens,
    tokenizer,
    sequence_col="sequences",
    filter_type=None,
    interleave_structure_sequence=False,
):
    if filter_type is None:
        return True

    sequences = example[sequence_col]
    max_seq_length = max(len(s) for s in sequences)

    def check_max_res_pos_in_seq():
        return any(len(s) <= tokenizer.max_res_pos_in_seq - 1 for s in sequences)

    def check_max_tokens():
        if max_tokens is None:
            return True

        effective_max_tokens = (
            max_tokens // 2 if interleave_structure_sequence else max_tokens
        )
        extra_tokens_per_document = tokenizer.num_start_tokens + (
            2 if interleave_structure_sequence else 1
        )

        return max_seq_length <= effective_max_tokens - extra_tokens_per_document

    filter_strategies = {
        "max_res_pos_in_seq": check_max_res_pos_in_seq,
        "max_tokens": check_max_tokens,
    }

    try:
        return filter_strategies[filter_type]()
    except KeyError:
        raise ValueError(f"Unknown length filter: {filter_type}")
