from typing import List, Optional

import numpy as np
import torch
from torch import stack
from transformers import PreTrainedTokenizerFast

from src.data.objects import ProteinDocument
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def get_flat_seq_pos_from_positions(
    positions,
    max_seq_pos: int = 1024,
    prepend_index=0,
    append_index=None,
    sep_index=None,
    num_start_tokens=1,
    num_end_tokens=1,
):
    # TODO: maybe raise exception if max_seq_pos exceeded rather than duplicating...
    if len(positions) > 0:
        flat_positions = [prepend_index] * num_start_tokens
        for sequence_positions in positions[:-1]:
            # add 1 so that sep doesnt have same position index
            flat_positions += [min(p + 1, max_seq_pos - 1) for p in sequence_positions]
            # default is to add position index for sep token from last sequence
            # this allows the model to predict sep token when shifting position indices in inputs
            flat_positions.append(sep_index or (sequence_positions[-1] + 1))
        flat_positions += [min(p + 1, max_seq_pos - 1) for p in positions[-1]]
        if append_index is None:
            # TODO: in case of multiple end tokens, we should prob restart...
            flat_positions += [
                sequence_positions[-1] + i for i in range(1, num_end_tokens + 1)
            ]
        else:
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
        append_index=None,
        sep_index=None,
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


def concatenate_pad_array(
    array_list,
    fill_value,
    num_start_tokens=1,
    num_end_tokens=1,
    pad_to_length: Optional[int] = None,
):
    if pad_to_length is not None:
        full_length = pad_to_length
    else:
        full_length = (
            sum(len(a) for a in array_list)
            + num_start_tokens
            + num_end_tokens
            + len(array_list)
            - 1  # sep tokens
        )
    if isinstance(array_list[0], list):
        full_array = np.full((full_length,), fill_value)
    else:
        assert isinstance(array_list[0], np.ndarray)
        full_array = np.full((full_length, *array_list[0].shape[1:]), fill_value)
    start_ix = num_start_tokens
    for arr in array_list:
        end_ix = start_ix + len(arr)
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
        add_document_token: bool = True,
        use_seq_pos: bool = False,
        max_seq_pos: int = 1024,
        max_tokens: Optional[int] = 5000,
        seq_struct_sep_token="[SEQ-STRUCT-SEP]",
        mask_below_plddt: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.add_bos_token = add_bos_token
        self.add_document_token = add_document_token
        self.num_start_tokens = int(self.add_bos_token) + int(self.add_document_token)
        self.use_seq_pos = use_seq_pos
        self.max_seq_pos = max_seq_pos
        self.max_tokens = max_tokens
        self.seq_struct_sep_token = seq_struct_sep_token
        self.mask_below_plddt = mask_below_plddt

        if not self.additional_special_tokens:
            additional_special_tokens = [
                tok.content
                for tok in self.added_tokens_decoder.values()
                if tok.special and tok.content not in self.special_tokens_map.values()
            ]
            self.add_special_tokens(
                {"additional_special_tokens": additional_special_tokens}
            )

    @property
    def seq_struct_sep_token_id(self):
        return self.convert_tokens_to_ids(self.seq_struct_sep_token)

    @property
    def aa_tokens(self):
        return self.convert_tokens_to_ids(list("ACDEFGHIKLMNPQRSTVWY"))

    def encode(
        self,
        proteins: ProteinDocument,
        document_token="[RAW]",
        padding="longest",
        max_length: Optional[int] = None,
        add_final_sep: bool = True,
        # TODO: allow custom fill value for coord / plddt padding?
        allow_unk: bool = False,
    ):
        """Encode a list of sequences into a single sequence of sequences tensor."""
        # TODO: add MSA / RAW document type token...
        concatenated_seqs = self.sep_token.join(proteins.sequences)
        if add_final_sep:
            concatenated_seqs += self.sep_token
        if self.add_bos_token:
            concatenated_seqs = self.bos_token + concatenated_seqs
        if document_token is not None:
            concatenated_seqs = document_token + concatenated_seqs
        else:
            assert (
                not self.add_document_token
            ), "Document type token expected but not provided"
        num_end_tokens = int(add_final_sep)
        tokenized = self(
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

        if not allow_unk:
            assert not (
                tokenized.input_ids == self.convert_tokens_to_ids("[UNK]")
            ).any(), "UNK tokens in input"
        if self.use_seq_pos:
            if proteins.positions is None:
                log.warning(
                    "Using seq_pos but positions not provided. Using default positions."
                )
                # +1 to match convert_sequence_with_positions
                # get_seq_pos_from_positions adds another offset
                positions = [list(range(1, len(seq) + 1)) for seq in proteins.sequences]
            else:
                positions = proteins.positions
            seq_pos = get_seq_pos_from_positions(
                tokenized.input_ids,
                positions,
                pad_token_id=self.pad_token_id,
                max_seq_pos=self.max_seq_pos,
                num_start_tokens=self.num_start_tokens,
                num_end_tokens=num_end_tokens,
            )
            tokenized.data["seq_pos"] = seq_pos
            assert seq_pos.shape[0] == tokenized.input_ids.shape[0]

        if proteins.backbone_coords is not None:
            tokenized.data["coords"] = torch.from_numpy(
                concatenate_pad_array(
                    proteins.backbone_coords,
                    fill_value=np.nan,
                    num_start_tokens=self.num_start_tokens,
                    num_end_tokens=num_end_tokens,
                    pad_to_length=tokenized.input_ids.shape[-1],
                )
            )
            assert (
                tokenized.data["coords"].shape[0] == tokenized.input_ids.shape[0]
            ), f"{tokenized.data['coords'].shape[0]} != {tokenized.input_ids.shape[0]}"

        tokenized.data["aa_mask"] = torch.isin(
            tokenized.input_ids, torch.tensor(self.aa_tokens)
        )
        if proteins.plddts is not None:
            tokenized.data["plddts"] = torch.from_numpy(
                concatenate_pad_array(
                    proteins.plddts,
                    fill_value=100.0,
                    num_start_tokens=self.num_start_tokens,
                    num_end_tokens=num_end_tokens,
                    pad_to_length=tokenized.input_ids.shape[-1],
                )
            )
            assert (
                tokenized.data["plddts"].shape[0] == tokenized.input_ids.shape[0]
            ), f"{tokenized.data['plddts'].shape[0]} != {tokenized.input_ids.shape[0]}"
            if self.mask_below_plddt is not None:
                # only mask structure tokens
                plddt_mask = (tokenized.data["plddts"] < self.mask_below_plddt) & ~(
                    tokenized.data["aa_mask"]
                )
                tokenized.data["plddt_mask"] = plddt_mask
                tokenized.data["input_ids"][plddt_mask] = self.mask_token_id
                tokenized.data["coords"][plddt_mask] = np.nan

        # TODO: handle nans
        # TODO: return sequence start and end positions?
        return tokenized

    def encode_completions(
        self,
        sequences,
        positions: Optional[List[int]] = None,
        bos_token="[SEP]",
        eos_token="[SEP]",
    ):
        assert isinstance(sequences, list)
        tokenized = self(
            [bos_token + seq + eos_token for seq in sequences],
            return_tensors="pt",
            padding="longest",
            truncation=False,
            add_special_tokens=False,
        )
        if self.use_seq_pos:
            all_positions = []
            for i, seq in enumerate(sequences):
                if positions is None:
                    seq_positions = [list(range(1, len(seq) + 1))]
                else:
                    seq_positions = positions[i]
                all_positions.append(
                    get_seq_pos_from_positions(
                        tokenized.input_ids[i],
                        seq_positions,
                        pad_token_id=self.pad_token_id,
                        max_seq_pos=self.max_seq_pos,
                        num_start_tokens=1
                        if bos_token
                        else 0,  # just bos_token (no document tag because we are completing prompt)
                        num_end_tokens=1 if eos_token else 0,
                    )
                )
            tokenized.data["seq_pos"] = stack(all_positions)

        return tokenized

    def decode_tokens(self, tokens):
        # TODO: some kind of assertion on shape
        assert tokens.ndim == 2
        dec = self.batch_decode(tokens)
        decoded_sequences = []

        for seq_of_seqs in dec:
            # we're trusting that [PAD] tokens are put in the correct place.
            decoded_seq_of_seqs = []
            for seq in seq_of_seqs.replace(" ", "").replace("[PAD]", "").split("[SEP]"):
                processed_seq = (
                    seq.replace("[RAW]", "")
                    .replace("[MSA]", "")
                    .replace("[start-of-document]", "")
                    .replace("[end-of-document]", "")
                )
                if processed_seq:
                    decoded_seq_of_seqs.append(processed_seq)
            assert decoded_seq_of_seqs, "Empty sequence"
            decoded_sequences.append(decoded_seq_of_seqs)
        if all(len(seq) == 1 for seq in decoded_sequences):
            decoded_sequences = [seq[0] for seq in decoded_sequences]
        return decoded_sequences
