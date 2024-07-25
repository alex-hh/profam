fpath = "/SAN/orengolab/cath_plm/ProFam/data/foldseek/failed_sequences.txt"
with open(fpath, 'r') as f:
    lines = f.readlines()
unique_lines = list(set(lines))

with open(fpath.replace(".txt", "unique.txt"), "w") as f:
    for line in unique_lines:
        f.write(line)