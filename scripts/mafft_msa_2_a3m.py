import sys, re

def read_fasta(stream):
    name, seq = None, []
    for line in stream:
        line = line.rstrip("\n")
        if not line: continue
        if line.startswith(">"):
            if name is not None:
                yield name, "".join(seq)
            name = line
            seq = []
        else:
            seq.append(line.strip())
    if name is not None:
        yield name, "".join(seq)

def to_a3m(master, seq):
    master = master.replace(".", "-")
    seq    = seq.replace(".", "-")
    out, ins = [], []
    for m, a in zip(master, seq):
        if m == "-":                            # column removed (master gap)
            if a != "-": ins.append(a.lower())  # becomes insertion
            continue
        if ins:                                  # flush pending insertions
            out.append("".join(ins).lower()); ins = []
        out.append("-" if a == "-" else a.upper())
    if ins: out.append("".join(ins).lower())
    return "".join(out)

def main():
    if len(sys.argv) != 3:
        sys.stderr.write(
            "Usage: python mafft_msa_2_a3m.py <input_aligned_fasta> <output_a3m>\n"
        )
        sys.exit(1)

    input_fasta = sys.argv[1]
    output_path = sys.argv[2]

    with open(input_fasta, "r") as f:
        recs = list(read_fasta(f))

    if not recs:
        sys.exit(0)

    master_name, master_seq = recs[0]
    master_seq = master_seq.replace(".", "-")

    # write master ungapped, others with lowercase insertions
    with open(output_path, "w") as out:
        out.write(f"{master_name}\n")
        out.write(re.sub(r"[-.\s]", "", master_seq).upper() + "\n")
        for name, seq in recs[1:]:
            out.write(f"{name}\n")
            out.write(to_a3m(master_seq, seq) + "\n")


if __name__ == "__main__":
    main()
