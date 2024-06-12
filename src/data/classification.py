import glob
import random
"""
For a given type of family
we create a dataset with a target family
and a mix of decoys (members of other families)
and non-decoys (members of the target family)
"""
import glob
import random
from datasets import Dataset
from transformers import PreTrainedTokenizerFast
from src.data import fasta
from src.data import utils as data_utils

def load_classifier_dataset(
    fasta_file_pattern,
    tokenizer: PreTrainedTokenizerFast,
    max_tokens_input=10000,
    max_seqs_to_predict=100,
    num_decoys_per_target=10,
    seed=42,
):
    paths = glob.glob(fasta_file_pattern)
    random.seed(seed)
    dataset_list = []

    for target_family_path in paths:
        # Load sequences from the target family
        labels, seqs = fasta.read_fasta(target_family_path)
        target_family_seqs = seqs.copy()

        # Sample sequences for the MSA (input_ids)
        msa_seqs = data_utils.sample_to_max_tokens(
            seqs, seed=seed, max_tokens=max_tokens_input
        )

        # Sample sequences for the targets (completion_ids)
        target_seqs = random.sample(target_family_seqs, max_seqs_to_predict)

        decoy_seqs = []
        for decoy_family_path in paths:
            if decoy_family_path == target_family_path:
                continue
            # Load sequences from the decoy family
            labels, seqs = fasta.read_fasta(decoy_family_path)
            # Sample sequences for the decoys
            decoy_seqs.extend(random.sample(seqs, num_decoys_per_target))

        # Combine target and decoy sequences
        completion_seqs = target_seqs + decoy_seqs
        random.shuffle(completion_seqs)

        # Create binary labels
        labels = [1] * len(target_seqs) + [0] * len(decoy_seqs)

        # Create a dictionary for the current target family
        family_dict = {
            "MSA": msa_seqs,
            "completion_seqs": completion_seqs,
            "labels": labels,
        }
        dataset_list.append(family_dict)

    # Create a dataset from the list of dictionaries
    dataset = Dataset.from_pandas(pd.DataFrame(dataset_list))

    # Tokenize the MSA and completion sequences
    dataset = dataset.map(
        lambda example: tokenize_msa(example, tokenizer, max_tokens=max_tokens_input),
        batched=False,
    )
    dataset = dataset.map(
        lambda example: tokenize_completions(example, tokenizer),
        batched=False,
        remove_columns=["MSA", "completion_seqs"],
    )

    # Set the format of the dataset
    dataset.set_format(
        type="torch", columns=["input_ids", "completion_ids", "labels"]
    )

    return dataset

def tokenize_msa(example, tokenizer: PreTrainedTokenizerFast, max_tokens=10000):
    concatenated_seqs = tokenizer.bos_token + tokenizer.sep_token.join(example["MSA"])
    tokenized = tokenizer(
        concatenated_seqs, return_tensors="pt", add_special_tokens=False
    )
    example["input_ids"] = tokenized.input_ids[0]
    return example

def tokenize_completions(example, tokenizer: PreTrainedTokenizerFast):
    example["completion_seqs"] = [
        tokenizer.sep_token + seq + tokenizer.sep_token
        for seq in example["completion_seqs"]
    ]
    tokenized = tokenizer(
        example["completion_seqs"],
        return_tensors="pt",
        padding="max_length",
        add_special_tokens=True,
    )
    example["completion_ids"] = tokenized.input_ids
    return example