import torch
from transformers import StoppingCriteria


def has_too_many_repeats(
    seq: str, repeat_length: int = 9, repeat_count: int = 9
) -> bool:
    """
    Heuristic to detect failed sampling by checking for repeated trailing substrings.
    Returns True if the last `repeat_length` chars appear at least `repeat_count` times in seq.
    """
    if len(seq) < repeat_length * repeat_count:
        return False

    substring = seq[-repeat_length:]
    # find all occurrences of the substring
    if seq.count(substring) >= repeat_count:
        return True
    return False


class RepeatStoppingCriteria(StoppingCriteria):
    def __init__(
        self, tokenizer, repeat_length=9, repeat_count=9, prompt_length: int = 0
    ):
        self.tokenizer = tokenizer
        self.repeat_length = repeat_length
        self.repeat_count = repeat_count
        self.prompt_length = int(prompt_length)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        # Consider only the portion generated beyond the original prompt
        if input_ids.ndim != 2 or input_ids.shape[0] == 0:
            return False
        generated_only = input_ids[0, self.prompt_length :]
        if generated_only.numel() == 0:
            return False
        seq = self.tokenizer.decode(generated_only, skip_special_tokens=True).replace(
            " ", ""
        )
        return has_too_many_repeats(seq, self.repeat_length, self.repeat_count)
