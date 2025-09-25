from src.models.inference import has_too_many_repeats
from src.sequence.fasta import fasta_generator

uniref90_path = "../data/uniref/uniref90.fasta"


def main():
    total_checked = 0
    total_flagged = 0
    report_every = 100_000

    for name, seq in fasta_generator(uniref90_path, keep_gaps=False, keep_insertions=True, to_upper=True):
        total_checked += 1
        if has_too_many_repeats(seq):
            total_flagged += 1
            print(f">{name}\n{seq}", flush=True)

        if total_checked % report_every == 0:
            proportion = total_flagged / total_checked if total_checked else 0.0
            print(
                f"Checked {total_checked} sequences; {total_flagged} flagged ({proportion:.4%}).",
                flush=True,
            )

    # Final report
    if total_checked:
        proportion = total_flagged / total_checked
        print(
            f"Done. Checked {total_checked} sequences; {total_flagged} flagged ({proportion:.4%}).",
            flush=True,
        )
    else:
        print("No sequences found.", flush=True)


if __name__ == "__main__":
    main()