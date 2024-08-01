import os
import sqlite3

db_path = f"/SAN/orengolab/cath_plm/profam_db/profam.db"
assert os.path.exists(db_path), "Profam database not found"
fasta_path = "/SAN/orengolab/cath_plm/ProFam/data/uniref100/uniref100.fasta"
assert os.path.exists(fasta_path), "Uniref100 fasta file not found"

conn = sqlite3.connect(db_path)
cursor = conn.cursor()
additional_seqs = 0


with open(fasta_path, "r") as file:
    current_id = None
    current_sequence = ""
    line_counter = 0
    for line in file:
        if line.startswith(">"):
            if current_id is not None:
                # check if the sequence is already in the database
                cursor.execute(
                    "SELECT sequence_id FROM sequences WHERE sequence_id = ?",
                    (current_id,),
                )
                # if the sequence is not in the database, add it
                if not cursor.fetchone():
                    cursor.execute(
                        "INSERT INTO sequences (sequence_id, sequence) VALUES (?, ?)",
                        (current_id, current_sequence),
                    )
                    additional_seqs += 1
            current_id = (
                line.strip().split()[0].split("UniRef100_")[1]
            )  # Extract the UniRef ID
            current_sequence = ""
        else:
            current_sequence += line.strip()
        line_counter += 1
        if line_counter % 10000 == 0:
            print(
                f"Processed {line_counter} lines, added {additional_seqs} additional sequences"
            )
    if current_id is not None:
        # check if the sequence is already in the database
        cursor.execute(
            "SELECT sequence_id FROM sequences WHERE sequence_id = ?", (current_id,)
        )
        # if the sequence is not in the database, add it
        if not cursor.fetchone():
            cursor.execute(
                "INSERT INTO sequences (sequence_id, sequence) VALUES (?, ?)",
                (current_id, current_sequence),
            )
            additional_seqs += 1

    conn.commit()
    conn.close()
    print(f"Added {additional_seqs} additional sequences to the database")
