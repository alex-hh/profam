from typing import List, Optional

from transformers import PreTrainedTokenizerFast

from src.data.utils import get_seq_pos_from_positions
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class ProFamTokenizer(PreTrainedTokenizerFast):
    """TODO: handle position encoding on here as well.
    (to make this really efficient we'd have to hack underlying rust code i think...)
    """

    def __init__(
        self,
        *args,
        add_bos_token: bool = True,
        add_document_type_token: bool = True,
        use_seq_pos: bool = False,
        max_seq_pos: int = 1024,
        max_tokens: Optional[int] = 5000,
        **kwargs,
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
            )
            tokenized.data["seq_pos"] = seq_pos
        # TODO: maybe return tokenized rather than just input ids.
        # then we can add position encoding to tokenized.data
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
