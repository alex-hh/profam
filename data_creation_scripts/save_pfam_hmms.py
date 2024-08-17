import argparse
from src.evaluators.hmmer import PFAMProfileHMM


def main(args):
    with open(args.accession_list_file, "r") as f:
        accessions = [line.strip() for line in f]
    evaluator = PFAMProfileHMM(
        "pfam",
        pfam_hmm_dir=args.pfam_hmm_dir,
        pfam_database=args.pfam_database,
    )
    for accession in accessions:
        print("Extracting HMM for", accession)
        evaluator.extract_hmm(accession, f"{args.output_dir}/{accession}.hmm")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "accession_list_file",
        type=str,
        help="File with list of PFAM accessions to extract",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory to save HMM files",
    )
    parser.add_argument(
        "--pfam_hmm_dir",
        type=str,
        default="../data/pfam/hmms",
        help="Directory to save HMM files",
    )
    parser.add_argument(
        "--pfam_database",
        type=str,
        default="../data/pfam/Pfam-A.hmm",
        help="PFAM database file",
    )

