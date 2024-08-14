from transformers import PreTrainedTokenizerFast


class ProFamTokenizer(PreTrainedTokenizerFast):
    def __init__(
        self,
        *args,
        num_start_tokens=2,
        add_final_sep: bool = True,
        add_bos_token: bool = True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.num_start_tokens = num_start_tokens
        self.add_bos_token = add_bos_token
        self.add_final_sep = add_final_sep

    def encode_sequences(self, sequences, document_type="[RAW]", padding="longest"):
        # TODO: add MSA / RAW document type token...
        concatenated_seqs = self.sep_token.join(sequences)
        if self.add_final_sep:
            concatenated_seqs += self.sep_token
        if self.add_bos_token:
            concatenated_seqs = self.bos_token + concatenated_seqs
        if document_type is not None:
            concatenated_seqs = document_type + concatenated_seqs
        tokenized = self.tokenizer(
            concatenated_seqs,
            truncation=False,  # shouldnt be necessary: bisection should handle
            return_tensors="pt",
            # padding="longest",
            padding=padding,
            add_special_tokens=False,
        )
        return tokenized.input_ids

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
