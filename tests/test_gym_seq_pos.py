import torch

from src.data.objects import ProteinDocument
from src.data.transforms import convert_sequences_adding_positions
from src.utils.tokenizers import ProFamTokenizer, get_seq_pos_from_positions

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


def test_prot_gym_pos_encoding():
    tokenizer = ProFamTokenizer(
        tokenizer_file="src/data/components/profam_tokenizer.json",
        unk_token="[UNK]",
        pad_token="[PAD]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        bos_token="[start-of-document]",
        add_special_tokens=True,
        add_final_sep=True,
        add_document_token=True,
        add_bos_token=True,
        use_seq_pos=True,
        max_seq_pos=1024,
        max_tokens=5000,
        seq_struct_sep_token="[SEQ-STRUCT-SEP]",
    )

    for case in test_cases_subs:
        # Process MSA sequences
        msa_proteins = ProteinDocument(sequences=case["msa_seqs"])
        msa_proteins = convert_sequences_adding_positions(
            msa_proteins,
            keep_gaps=case["keep_gaps"],
            keep_insertions=True,
            to_upper=True,
            use_msa_pos=True,
        )

        # Process completion sequences
        completion_proteins = ProteinDocument(sequences=case["completion_seqs"])
        completion_proteins = convert_sequences_adding_positions(
            completion_proteins,
            keep_gaps=case["keep_gaps"],
            keep_insertions=True,
            to_upper=True,
            use_msa_pos=True,
        )

        # Tokenize MSA
        msa_tokenized = tokenizer.encode(
            msa_proteins,
            document_token="[MSA]",
            add_final_sep=False,
        )

        completion_tokenized = tokenizer.encode_completions(
            completion_proteins.sequences,
            bos_token=tokenizer.sep_token,
        )

        msa_seq_pos = get_seq_pos_from_positions(
            msa_tokenized.input_ids,
            msa_proteins.positions,
            pad_token_id=tokenizer.pad_token_id,
            max_seq_pos=tokenizer.max_seq_pos,
            num_start_tokens=tokenizer.num_start_tokens,
            num_end_tokens=0,  # No end tokens for MSA
        )

        # Check MSA positions
        assert (
            msa_seq_pos == torch.tensor(case["msa_pos"])
        ).all(), (
            f"MSA positions mismatch: {msa_proteins.positions} != {case['msa_pos']}"
        )

        for i, comp in enumerate(case["completion_pos"]):
            assert (
                completion_tokenized.seq_pos[i] == torch.tensor(comp)
            ).all(), f"Completion positions mismatch: {completion_tokenized.seq_pos[i]} != {comp}"

        assert (
            completion_tokenized.seq_pos == torch.tensor(case["completion_pos"])
        ).all(), f"Completion positions mismatch: {completion_tokenized.seq_pos} != {case['completion_pos']}"


print("Running positional embedding tests...")
test_prot_gym_pos_encoding()
print("All tests passed!")
