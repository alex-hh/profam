from typing import List, Optional

import numpy as np
import torch
from transformers import PreTrainedTokenizerFast

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def get_flat_seq_pos_from_positions(
    positions,
    max_seq_pos: int = 1024,
    prepend_index=0,
    append_index=0,
    sep_index=0,
    num_start_tokens=1,
    num_end_tokens=1,
):
    # TODO: maybe raise exception if max_seq_pos exceeded rather than duplicating...
    if len(positions) > 0:
        flat_positions = [prepend_index] * num_start_tokens
        for sequence_positions in positions[:-1]:
            # add 1 so that sep doesnt have same position index
            flat_positions += [min(p + 1, max_seq_pos - 1) for p in sequence_positions]
            flat_positions.append(sep_index)
        flat_positions += [min(p + 1, max_seq_pos - 1) for p in positions[-1]]
        flat_positions += [append_index] * num_end_tokens
        return flat_positions
    else:
        return []


def get_seq_pos_from_positions(
    input_ids,
    positions,
    pad_token_id,
    max_seq_pos: int = 1024,
    num_start_tokens=1,
    num_end_tokens=1,
):
    assert input_ids.ndim == 1
    seq_pos = torch.zeros_like(input_ids)
    # TODO: convert to array and use concatenate_pad_array instead
    flat_pos = get_flat_seq_pos_from_positions(
        positions,
        max_seq_pos=max_seq_pos,
        prepend_index=0,
        append_index=0,
        sep_index=0,
        num_start_tokens=num_start_tokens,  # TODO: handle better
        num_end_tokens=num_end_tokens,
    )
    pad_any = torch.argwhere(input_ids == pad_token_id)
    if pad_any.any():
        pad_start = pad_any.min()
    else:
        pad_start = input_ids.shape[0]
    seq_pos[:pad_start] = torch.tensor(flat_pos)
    return seq_pos


def concatenate_pad_array(array_list, fill_value, num_start_tokens=1, num_end_tokens=1):
    full_length = sum(len(a) for a in array_list) + num_start_tokens + num_end_tokens
    if isinstance(array_list[0], list):
        full_array = np.full((full_length,), fill_value)
    else:
        assert isinstance(array_list[0], np.ndarray)
        full_array = np.full((full_length, *array_list[0].shape[1:]), fill_value)
    start_ix = num_start_tokens
    for arr in array_list:
        end_ix = start_ix + arr.shape[0]
        full_array[start_ix:end_ix] = arr
        start_ix = end_ix + 1  # +1 for sep token
    return full_array


class ProFamTokenizer(PreTrainedTokenizerFast):
    """TODO: handle position encoding on here as well.
    (to make this really efficient we'd have to hack underlying rust code i think...)
    """

    # TODO: handle max tokens?
    def __init__(
        self,
        *args,
        add_bos_token: bool = True,
        add_document_type_token: bool = True,
        use_seq_pos: bool = False,
        max_seq_pos: int = 1024,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.add_bos_token = add_bos_token
        self.add_document_type_token = add_document_type_token
        self.num_start_tokens = int(self.add_bos_token) + int(
            self.add_document_type_token
        )
        self.use_seq_pos = use_seq_pos
        self.max_seq_pos = max_seq_pos
        self.max_tokens = max_tokens

    def encode_sequences(
        self,
        sequences,
        positions: Optional[List[int]] = None,
        document_type="[RAW]",
        padding="longest",
        max_length: Optional[int] = None,
        add_final_sep: bool = True,
        coords: Optional[List[np.ndarray]] = None,
        plddts: Optional[List[np.ndarray | List]] = None,
    ):
        # TODO: add MSA / RAW document type token...
        concatenated_seqs = self.sep_token.join(sequences)
        if add_final_sep:
            concatenated_seqs += self.sep_token
        if self.add_bos_token:
            concatenated_seqs = self.bos_token + concatenated_seqs
        if document_type is not None:
            concatenated_seqs = document_type + concatenated_seqs
        else:
            assert (
                not self.add_document_type_token
            ), "Document type token expected but not provided"
        num_end_tokens = int(add_final_sep)
        tokenized = self.tokenizer(
            concatenated_seqs,
            truncation=False,  # shouldnt be necessary: bisection should handle
            return_tensors="pt",
            # padding="longest",
            padding=padding,
            add_special_tokens=False,
            max_length=max_length,
        )
        if self.max_tokens is not None:
            assert tokenized.input_ids.shape[1] <= self.max_tokens, (
                tokenized.input_ids.shape[1],
                self.max_tokens,
            )
        tokenized.data = {k: v.squeeze() for k, v in tokenized.data.items()}
        assert tokenized.input_ids.ndim == 1
        if self.use_seq_pos:
            if positions is None:
                log.warning(
                    "Using seq_pos but positions not provided. Using default positions."
                )
                # +1 to match convert_sequence_with_positions
                # get_seq_pos_from_positions adds another offset
                positions = [list(range(1, len(seq) + 1)) for seq in sequences]
            seq_pos = get_seq_pos_from_positions(
                tokenized.input_ids,
                positions,
                pad_token_id=self.pad_token_id,
                max_seq_pos=self.max_seq_pos,
                num_start_tokens=self.num_start_tokens,
                num_end_tokens=num_end_tokens,
            )
            tokenized.data["seq_pos"] = seq_pos
            assert seq_pos.shape[0] == tokenized.input_ids[0]

        if coords is not None:
            tokenized.data["coords"] = torch.from_numpy(
                concatenate_pad_array(
                    coords,
                    fill_value=np.nan,
                    num_start_tokens=self.num_start_tokens,
                    num_end_tokens=num_end_tokens,
                )
            )
            assert tokenized.data["coords"].shape[0] == tokenized.input_ids[0]

        if plddts is not None:
            tokenized.data["plddts"] = torch.from_numpy(
                concatenate_pad_array(
                    plddts,
                    fill_value=np.nan,
                    num_start_tokens=self.num_start_tokens,
                    num_end_tokens=num_end_tokens,
                )
            )
            assert tokenized.data["plddts"].shape[0] == tokenized.input_ids[0]

        # TODO: handle nans
        # TODO: return sequence start and end positions?
        return tokenized

    def decode_tokens(self, tokens):
        # TODO: some kind of assertion on shape
        assert tokens.ndim == 2 and tokens.shape[0] == 1
        if tokens[:, -1] == self.sep_token_id:
            tokens = tokens[:, :-1]
        # TODO: use batch_decode for batches
        dec = self.decode(tokens.squeeze(0))
        return [
            s.replace("[RAW]", "")
            .replace("[MSA]", "")
            .replace("[start-of-document]", "")
            .replace("[end-of-document]", "")
            for s in dec.replace(" ", "").split("[SEP]")
        ]
