from functools import partial

import numpy as np

from src.data.objects import ProteinDocument
from src.data.processors.transforms import (
    convert_aligned_sequence_adding_positions,
    preprocess_aligned_sequences_sampling_to_max_tokens,
)
from src.data.tokenizers import get_residue_index_from_positions

"""
replicates the pre-processing and
seq encoding used for proteinGym
tests not yet implemented for indels
indels still not currently handled
correctly in Gym
"""
# fmt: off
test_cases_subs = [
    {
        "msa_seqs": ["ACD", "ACD"],
        "completion_seqs": ["ACD", "ACD"],
        "msa_pos": [0, 0, 2, 3, 4, 0, 2, 3, 4],
        "completion_pos": [[0, 2, 3, 4, 0], [0, 2, 3, 4, 0]],
        "keep_gaps": False,
    },
    {
        "msa_seqs": [".ACDE", "aACD-"],
        "completion_seqs": ["ACD", "ACD"],
        "msa_pos": [0, 0, 2, 3, 4, 5, 0, 1, 2, 3, 4],
        "completion_pos": [[0, 2, 3, 4, 0], [0, 2, 3, 4, 0]],
        "keep_gaps": False,
    },
    {
        "msa_seqs": [".ACDE", "aACD-"],
        "completion_seqs": ["ACD", "ACD"],
        "msa_pos": [0, 0, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5],
        "completion_pos": [[0, 2, 3, 4, 0], [0, 2, 3, 4, 0]],
        "keep_gaps": True,
    },
]

test_cases_indels = [
    # todo not implemented yet - tests are WIP
    {
        "msa_seqs": ["GAPGAPGAP", "--GIRF-G-", "--G-GF-G-"],
        "completion_seqs": [
            "GAPGAPGAP",
            "--GIRF-G-",
            "--G-GF-G-",
        ],
        "msa_pos": [
            0, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            0, 0, 4, 5, 6, 7, 9,
            0, 4, 6, 7, 9,
        ],
        "completion_pos": [
            [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0],
            [0, 0, 4, 5, 6, 7, 9, 0, 0, 0, 0],
            [0, 4, 6, 7, 9, 0, 0, 0, 0, 0, 0],
        ],
        "keep_gaps": False,
    },
    {
        "msa_seqs": ["--GIRF-G-", "--G-GF-G-"],
        "completion_seqs": [
            "--GIRF-G-",
            "--G-GF-G-",
        ],
        "msa_pos": [[0, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
        "completion_pos": [[1, 2, 3, 4, 5, 6], [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
        "keep_gaps": True,
    },
    {
        "msa_seqs": ["GASGASG", "FDD.sN.", "-VTrnD."],
        "completion_seqs": ["GASGASGA", "FDD.sN..", "-VTrnD.."],
        "msa_pos": [0, 0, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        "completion_pos": [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
        "keep_gaps": True,
    },
]
# fmt: on


def test_prot_gym_pos_encoding(profam_tokenizer):
    for case in test_cases_subs:
        # Process MSA sequences
        msa_proteins = ProteinDocument(sequences=case["msa_seqs"])
        sequence_converter = partial(
            convert_aligned_sequence_adding_positions,
            keep_gaps=case["keep_gaps"],
            keep_insertions=True,
            to_upper=True,
            use_msa_pos=True,
        )
        msa_proteins = preprocess_aligned_sequences_sampling_to_max_tokens(
            msa_proteins,
            tokenizer=profam_tokenizer,
            sequence_converter=sequence_converter,
            shuffle=False,
            max_tokens=None,
        )

        # Process completion sequences
        completion_proteins = ProteinDocument(sequences=case["completion_seqs"])
        completion_proteins = preprocess_aligned_sequences_sampling_to_max_tokens(
            completion_proteins,
            tokenizer=profam_tokenizer,
            sequence_converter=sequence_converter,
            shuffle=False,
            max_tokens=None,
        )

        # Tokenize MSA
        msa_tokenized = profam_tokenizer.encode(
            msa_proteins,
            document_token="[MSA]",
            add_final_sep=False,
        )

        completion_tokenized = profam_tokenizer.encode_completions(
            completion_proteins.sequences,
            bos_token=profam_tokenizer.sep_token,
        )

        msa_seq_pos = get_residue_index_from_positions(
            msa_tokenized.input_ids,
            msa_proteins.residue_positions,
            pad_token_id=profam_tokenizer.pad_token_id,
            max_res_pos_in_seq=profam_tokenizer.max_res_pos_in_seq,
            num_start_tokens=profam_tokenizer.num_start_tokens,
            num_end_tokens=0,  # No end tokens for MSA
        )

        # Check MSA positions
        assert (
            msa_seq_pos == np.array(case["msa_pos"])
        ).all(), (
            f"MSA positions mismatch: {msa_proteins.positions} != {case['msa_pos']}"
        )

        for i, comp in enumerate(case["completion_pos"]):
            assert (
                completion_tokenized.residue_index[i] == np.array(comp)
            ).all(), f"Completion positions mismatch: {completion_tokenized.residue_index[i]} != {comp}"

        assert (
            completion_tokenized.residue_index == np.array(case["completion_pos"])
        ).all(), f"Completion positions mismatch: {completion_tokenized.residue_index} != {case['completion_pos']}"
