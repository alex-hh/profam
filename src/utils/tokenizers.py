from transformers import PreTrainedTokenizerFast


class ProFamTokenizer(PreTrainedTokenizerFast):
    def encode_sequences(self, sequences):
        # TODO: add MSA / RAW document type token...
        concatenated_seqs = (
            self.bos_token + self.sep_token.join(sequences) + self.sep_token
        )
        tokenized = self.tokenizer(
            concatenated_seqs,
            truncation=False,  # shouldnt be necessary: bisection should handle
            return_tensors="pt",
            # padding="longest",
            padding="longest",
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
