from typing import List, Optional

import numpy as np
from transformers import PreTrainedTokenizerFast

from src.data.objects import ProteinDocument
from src.data.utils import examples_list_to_dict
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def concatenate_pad_array(
    array_list,
    fill_value,
    num_start_tokens=1,
    num_end_tokens=1,
    pad_to_length: Optional[int] = None,
):
    arrays_length = (
        sum(len(a) for a in array_list)
        + num_start_tokens
        + num_end_tokens
        + len(array_list)
        - 1  # sep tokens
    )
    if pad_to_length is not None:
        full_length = pad_to_length
        assert arrays_length <= full_length, f"{arrays_length} > {full_length}"
    else:
        full_length = arrays_length
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


def get_sequence_of_sequences(
    proteins: ProteinDocument,
    sep_token: str = "[SEP]",
    bos_token: Optional[str] = None,
    add_final_sep: bool = True,
    document_token: Optional[str] = "[RAW]",
):
    concatenated_seqs = sep_token.join(proteins.sequences)
    if add_final_sep:
        concatenated_seqs += sep_token
    if document_token is not None:
        concatenated_seqs = document_token + concatenated_seqs
    if bos_token is not None:
        concatenated_seqs = bos_token + concatenated_seqs
    return concatenated_seqs


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
        seq_struct_sep_token="|",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.add_bos_token = add_bos_token
        self.add_document_token = add_document_token
        self.seq_struct_sep_token = seq_struct_sep_token

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

    @property
    def num_start_tokens(self):
        return int(self.add_bos_token) + int(self.add_document_token)

    def encode(
        self,
        proteins: ProteinDocument,
        document_token: Optional[str] = "[RAW]",
        padding="longest",
        max_length: Optional[int] = None,
        add_final_sep: bool = True,
        allow_unk: bool = False,
    ):
        """Encode a list of sequences into a single sequence of sequences tensor."""
        # TODO: add MSA / RAW document type token...
        if self.add_document_token:
            assert document_token is not None, "Document type token expected"
        concatenated_seqs = get_sequence_of_sequences(
            proteins,
            sep_token=self.sep_token,
            bos_token=self.bos_token if self.add_bos_token else None,
            add_final_sep=add_final_sep,
            document_token=document_token,
        )
        num_end_tokens = int(add_final_sep)
        tokenized = self(
            concatenated_seqs,
            truncation=False,  # shouldnt be necessary: bisection should handle
            return_tensors="np",  # https://huggingface.co/docs/datasets/nlp_process#map
            # padding="longest",
            padding=padding,
            add_special_tokens=False,
            max_length=max_length,
            return_token_type_ids=False,
        )
        tokenized.data = {k: v.squeeze(0) for k, v in tokenized.data.items()}
        assert tokenized.input_ids.ndim == 1

        if not allow_unk:
            assert not (
                tokenized.input_ids == self.convert_tokens_to_ids("[UNK]")
            ).any(), "UNK tokens in input"

        else:
            # TODO: handle more carefully
            tokenized.data["aa_mask"] = np.ones_like(tokenized.input_ids).astype(bool)
            if proteins.backbone_coords is not None:
                tokenized.data["structure_mask"] = tokenized.data["coords_mask"].any(
                    axis=(-1, -2)
                )
            else:
                tokenized.data["structure_mask"] = np.zeros_like(
                    tokenized.input_ids
                ).astype(bool)

        if proteins.original_size is not None:
            tokenized.data["original_size"] = proteins.original_size

        if proteins.identifier is not None:
            tokenized.data["identifier"] = proteins.identifier

        return tokenized

    def batched_encode(
        self,
        proteins_list: List[ProteinDocument],
        document_token="[RAW]",
        padding="longest",
        max_length: Optional[int] = None,
        add_final_sep: bool = True,
        allow_unk: bool = False,
        actually_batched: bool = False,
    ):
        if actually_batched:
            raise NotImplementedError("Actually batched encoding not implemented yet")

        return examples_list_to_dict(
            [
                self.encode(
                    proteins,
                    document_token=document_token,
                    padding=padding,
                    max_length=max_length,
                    add_final_sep=add_final_sep,
                    allow_unk=allow_unk,
                )
                for proteins in proteins_list
            ]
        )

    def encode_completions(
        self,
        sequences,
        bos_token="[SEP]",
        eos_token="[SEP]",
    ):
        assert isinstance(sequences, list)
        sequences_w_sp_tokens = [bos_token + seq + eos_token for seq in sequences]
        tokenized = self(
            sequences_w_sp_tokens,
            return_tensors="np",
            padding="longest",
            truncation=False,
            add_special_tokens=False,
        )

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
