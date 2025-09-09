"""
Takes the generated fasta file for all of the funfams and foldseek families
and takes the first generated sequence for each family
and places it in a new fasta file with format:
{funfams/foldseek}_{val/test}_{ensemble/single}_{fam_id}_gen0.fasta
all files should be placed in the same directory
there should also be a file called fasta_file_list.txt

which has one filename per line

current paths to fastas:
  "../sampling_results/foldseek_val/sampler=ensemble_tp=0.95_ns=20_nv=8_red=mean_probs/*.fasta"
  "../sampling_results/foldseek_val/sampler=single_tp=0.95_ns=20_nv=8_red=mean_probs/*.fasta"
    "../sampling_results/funfams_val/sampler=ensemble_tp=0.95_ns=20_nv=8_red=mean_probs/*.fasta"
  "../sampling_results/funfams_val/sampler=single_tp=0.95_ns=20_nv=8_red=mean_probs/*.fasta"

    "../sampling_results/foldseek_test/sampler=ensemble_tp=0.95_ns=20_nv=8_red=mean_probs/*.fasta"
  "../sampling_results/foldseek_test/sampler=single_tp=0.95_ns=20_nv=8_red=mean_probs/*.fasta"
    "../sampling_results/funfams_test/sampler=ensemble_tp=0.95_ns=20_nv=8_red=mean_probs/*.fasta"
  "../sampling_results/funfams_test/sampler=single_tp=0.95_ns=20_nv=8_red=mean_probs/*.fasta"

ensure that the ordering in the fasta file list adds each file in the order above
"""

import argparse
import glob
import os
import re
import sys
from typing import List, Tuple

try:
    # Local import from repo
    from src.sequence.fasta import first_sequence, output_fasta
except Exception as e:
    first_sequence = None  # type: ignore
    output_fasta = None  # type: ignore


DEFAULT_PATTERNS_IN_ORDER: List[str] = [
    "../sampling_results/foldseek_val/sampler=ensemble_tp=0.95_ns=20_nv=8_red=mean_probs/*.fasta",
    "../sampling_results/foldseek_val/sampler=single_tp=0.95_ns=20_nv=8_red=mean_probs/*.fasta",
    "../sampling_results/funfams_val/sampler=ensemble_tp=0.95_ns=20_nv=8_red=mean_probs/*.fasta",
    "../sampling_results/funfams_val/sampler=single_tp=0.95_ns=20_nv=8_red=mean_probs/*.fasta",
    "../sampling_results/foldseek_test/sampler=ensemble_tp=0.95_ns=20_nv=8_red=mean_probs/*.fasta",
    "../sampling_results/foldseek_test/sampler=single_tp=0.95_ns=20_nv=8_red=mean_probs/*.fasta",
    "../sampling_results/funfams_test/sampler=ensemble_tp=0.95_ns=20_nv=8_red=mean_probs/*.fasta",
    "../sampling_results/funfams_test/sampler=single_tp=0.95_ns=20_nv=8_red=mean_probs/*.fasta",
]


def parse_metadata_from_path(input_fasta_path: str) -> Tuple[str, str, str, str]:
    """Extract (dataset, split, sampler, fam_id) from an input FASTA path.

    - dataset: one of {foldseek, funfams}
    - split: one of {val, test}
    - sampler: one of {ensemble, single}
    - fam_id: derived from the FASTA filename stem
    """
    dataset = "unknown"
    split = "unknown"
    sampler = "unknown"

    # Try to infer from expected directory structure
    m = re.search(
        r"sampling_results/(foldseek|funfams)_(val|test)/sampler=(ensemble|single)",
        input_fasta_path,
    )
    if m:
        dataset, split, sampler = m.group(1), m.group(2), m.group(3)

    fam_id = os.path.splitext(os.path.basename(input_fasta_path))[0]
    fam_id = re.sub(r"[^A-Za-z0-9_.\-]+", "_", fam_id)

    return dataset, split, sampler, fam_id


def make_output_filename(dataset: str, split: str, sampler: str, fam_id: str) -> str:
    return f"{dataset}_{split}_{sampler}_{fam_id}_gen0.fasta"


def ensure_imports_available() -> None:
    if first_sequence is None or output_fasta is None:
        print(
            "Error: Failed to import FASTA utilities from src.sequence.fasta.",
            file=sys.stderr,
        )
        sys.exit(1)


def process_inputs(
    patterns_in_order: List[str], output_dir: str, overwrite: bool = False
) -> List[str]:
    """Process input FASTA files and return the list of created output paths.

    The returned list preserves group ordering as provided by patterns_in_order
    and uses lexicographic ordering within each group.
    """
    ensure_imports_available()

    os.makedirs(output_dir, exist_ok=True)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    created_outputs: List[str] = []
    seen_inputs = set()

    for rel_pattern in patterns_in_order:
        input_fastas = glob.glob(rel_pattern)

        if not input_fastas:
            # Be quiet if pattern simply doesn't match; just continue
            continue

        for fasta_path in input_fastas:
            if fasta_path in seen_inputs:
                continue

            dataset, split, sampler, fam_id = parse_metadata_from_path(fasta_path)

            try:
                _header, seq = first_sequence(fasta_path)  # type: ignore[arg-type]
            except Exception:
                print(
                    f"Warning: could not read first sequence from {fasta_path}; skipping.",
                    file=sys.stderr,
                )
                continue

            out_name = make_output_filename(dataset, split, sampler, fam_id)
            out_path = os.path.abspath(os.path.join(output_dir, out_name))

            if overwrite or not os.path.exists(out_path):
                try:
                    output_fasta(["gen0"], [seq], out_path)  # type: ignore[arg-type]
                except Exception as e:
                    print(
                        f"Error writing output FASTA {out_path}: {e}", file=sys.stderr
                    )
                    continue

            created_outputs.append(out_path)
            seen_inputs.add(fasta_path)

    return created_outputs


def write_list_file(output_dir: str, fasta_paths: List[str]) -> str:
    list_path = os.path.abspath(os.path.join(output_dir, "fasta_file_list.txt"))
    with open(list_path, "w") as f:
        for p in fasta_paths:
            f.write(p + "\n")
    return list_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the first generated sequence per family and write one FASTA "
            "per family into an output directory, along with an ordered list file."
        )
    )
    parser.add_argument(
        "--output_dir",
        default="../sampling_results/funfam_foldseek_gen0_combined",
        help="Directory to write per-family FASTAs and fasta_file_list.txt",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing per-family FASTAs if present",
    )
    parser.add_argument(
        "--patterns_file",
        default=None,
        help=(
            "Optional path to a text file containing one glob pattern per line, "
            "in the desired processing order. If not provided, built-in defaults "
            "from the module docstring are used."
        ),
    )
    return parser


def load_patterns_from_file(patterns_file: str) -> List[str]:
    patterns: List[str] = []
    with open(patterns_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            patterns.append(line)
    return patterns


def main(argv: List[str]) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)

    patterns_in_order = (
        load_patterns_from_file(args.patterns_file)
        if args.patterns_file
        else DEFAULT_PATTERNS_IN_ORDER
    )

    created = process_inputs(patterns_in_order, args.output_dir, args.overwrite)
    list_file = write_list_file(args.output_dir, created)

    print(
        f"Wrote {len(created)} FASTA files to {os.path.abspath(args.output_dir)}\n"
        f"List file: {list_file}"
    )
    if len(created) == 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
