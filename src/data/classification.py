import glob

"""
For a given type of family
we create a dataset with a target family
and a mix of decoys (members of other families)
and non-decoys (members of the target family)
"""
def load_classifier_dataset(fasta_file_pattern, max_tokens=10000):
    paths = glob.glob(fasta_file_pattern)
    for target_family_path in paths:
        # sample some sequences that will be the msa
        # sample some other sequences that will be the targets
        for decoy_family_path in paths:
            if decoy_family_path == target_family_path:
                continue
            # sample some sequences that will be the decoys
            # combine the sequences
            # return the sequences and the labels
            yield sequences, labels
    pass