from src.data.processors.transforms import (
    convert_aligned_sequence_adding_positions,
    convert_raw_sequence_adding_positions,
)


def test_convert_sequence_with_positions():
    examples = [
        {
            "seq": "ABC",
            "keep_gaps": True,
            "keep_insertions": True,
            "to_upper": True,
            "use_msa_pos": True,
            "expected": ["ABC", [1, 2, 3], [True, True, True]],
        },
        {
            "seq": "ABC",
            "keep_gaps": True,
            "keep_insertions": True,
            "to_upper": True,
            "use_msa_pos": False,
            "expected": ["ABC", [1, 2, 3], [True, True, True]],
        },
        {
            "seq": "aBC",
            "keep_gaps": True,
            "keep_insertions": True,
            "to_upper": True,
            "use_msa_pos": True,
            "expected": ["ABC", [0, 1, 2], [False, True, True]],
        },
        {
            "seq": "aBC",
            "keep_gaps": True,
            "keep_insertions": True,
            "to_upper": True,
            "use_msa_pos": False,
            "expected": ["ABC", [1, 2, 3], [False, True, True]],
        },
        {
            "seq": "aBCaa",
            "keep_gaps": True,
            "keep_insertions": True,
            "to_upper": True,
            "use_msa_pos": True,
            "expected": ["ABCAA", [0, 1, 2, 2, 2], [False, True, True, False, False]],
        },
        {
            "seq": "azBCaa",
            "keep_gaps": True,
            "keep_insertions": True,
            "to_upper": True,
            "use_msa_pos": True,
            "expected": [
                "AZBCAA",
                [0, 0, 1, 2, 2, 2],
                [False, False, True, True, False, False],
            ],
        },
        {
            "seq": "azBzCaa",
            "keep_gaps": True,
            "keep_insertions": False,
            "to_upper": False,
            "use_msa_pos": True,
            "expected": ["BC", [1, 2], [True, True]],
        },
        {
            "seq": "abCdEFgHi",
            "keep_gaps": False,
            "keep_insertions": True,
            "to_upper": True,
            "use_msa_pos": False,
            "expected": [
                "ABCDEFGHI",
                [1, 2, 3, 4, 5, 6, 7, 8, 9],
                [False, False, True, False, True, True, False, True, False],
            ],
        },
        {
            "seq": "..--ab-C.d-E..f-G.-",
            "keep_gaps": False,
            "keep_insertions": True,
            "to_upper": True,
            "use_msa_pos": True,
            "expected": [
                "ABCDEFG",
                [2, 2, 4, 4, 6, 6, 8],
                [False, False, True, False, True, False, True],
            ],
        },
        {
            "seq": "..--ab-C.d-E..f-G.-",
            "keep_gaps": False,
            "keep_insertions": True,
            "to_upper": True,
            "use_msa_pos": False,
            "expected": ["ABCDEFG", [1, 2, 3, 4, 5, 6, 7], None],
        },
    ]
    for i, d in enumerate(examples):
        seq = d["seq"]
        keep_gaps = d["keep_gaps"]
        keep_insertions = d["keep_insertions"]
        to_upper = d["to_upper"]
        use_msa_pos = d["use_msa_pos"]
        expected = d["expected"]
        sequence, positions, is_match = convert_aligned_sequence_adding_positions(
            seq,
            keep_gaps=keep_gaps,
            keep_insertions=keep_insertions,
            to_upper=to_upper,
            use_msa_pos=use_msa_pos,
        )
        if sequence != expected[0]:
            raise ValueError(
                f"Test {i} failed. Expected {expected[0]} but got {sequence}"
            )
        if str(positions) != str(expected[1]):
            raise ValueError(
                f"Test {i} failed. Expected {expected[1]} but got {positions}"
            )
        if expected[2] is not None:
            if str(is_match) != str(expected[2]):
                raise ValueError(
                    f"Test {i} failed. Expected {expected[2]} but got {is_match}"
                )
