import sqlite3

def create_database(db_file):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sequences (
            uniref_id TEXT PRIMARY KEY,
            sequence TEXT
        )
    ''')

    conn.commit()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS families (
            family_id TEXT PRIMARY KEY,
            family_type TEXT,
            family_name TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Function to parse the UniRef100 FASTA file and insert sequences into the database
def insert_sequences(db_file, fasta_file):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    with open(fasta_file, 'r') as file:
        current_id = None
        current_sequence = ''
        line_counter = 0
        for line in file:
            if line.startswith('>'):
                if current_id is not None:
                    cursor.execute('INSERT OR REPLACE INTO sequences VALUES (?, ?)', (current_id, current_sequence))
                current_id = line.strip().split()[0].split('UniRef100_')[1]  # Extract the UniRef ID
                current_sequence = ''
            else:
                current_sequence += line.strip()
            line_counter += 1
            if line_counter % 100000 == 0:
                print(f"Processed {line_counter} lines")
        if current_id is not None:
            cursor.execute('INSERT OR REPLACE INTO sequences VALUES (?, ?)', (current_id, current_sequence))

    conn.commit()
    conn.close()

# Function to insert families into the database
def insert_families(family_type, family_file):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    with open(family_file, 'r') as file:
        for line in file:
            family_id, family_name = line.strip().split()
            cursor.execute('INSERT OR REPLACE INTO families VALUES (?, ?, ?)', (family_id, family_type, family_name))
    conn.commit()

# Function to insert protein-family associations into the database
def insert_protein_families(family_type, association_file):
    with open(association_file, 'r') as file:
        for line in file:
            if family_type == 'CATH':
                uniref_id, family_id, domain_start, domain_end = line.strip().split()
                cursor.execute('INSERT OR REPLACE INTO protein_families VALUES (?, ?, ?, ?)',
                               (uniref_id, family_id, domain_start, domain_end))
            else:
                uniref_id, family_id = line.strip().split()
                cursor.execute('INSERT OR REPLACE INTO protein_families VALUES (?, ?, NULL, NULL)',
                               (uniref_id, family_id))
    conn.commit()


if __name__=="__main__":
    db_file = 'uniref100.fasta'
    create_database(db_file)
    fasta_file = '/SAN/orengolab/cath_plm/ProFam/data/uniref100/uniref100.fasta'
    insert_sequences(db_file, fasta_file)
    # Parse and insert data into the database
    insert_families('CATH', 'cath_superfamilies.txt')
    insert_families('Pfam', 'pfam_families.txt')
    insert_families('EC', 'ec_numbers.txt')
    insert_protein_families('CATH', 'cath_associations.txt')
    insert_protein_families('Pfam', 'pfam_associations.txt')
    insert_protein_families('EC', 'ec_associations.txt')