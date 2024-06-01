import sqlite3
import time

"""
Sample of uniref100.fasta:
>UniRef100_Q6GZX4 Putative transcription factor 001R n=4 Tax=Ranavirus TaxID=10492 RepID=001R_FRG3G
MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVEWNNPPS
EKGLIVGHFSGIKYKGEKAQASEVDVNKMCCWVSKFKDAMRRYQGIQTCKIPGKVLSDLD
AKIKAYNLTVEGVEGFVRYSRVTKQHVAAFLKELRHSKQYENVNLIHYILTDKRVDIQHL
EKDLVKDFKALVESAHRMRQGHMINVKYILYQLLKKHGHGPDGPDILTVKTGSKGVLYDD
SFRKIYTDLGWKFTPL
>UniRef100_Q6GZX3 Uncharacterized protein 002L n=3 Tax=Frog virus 3 TaxID=10493 RepID=002L_FRG3G
MSIIGATRLQNDKSDTYSAGPCYAGGCSAFTPRGTCGKDWDLGEQTCASGFCTSQPLCAR
IKKTQVCGLRYSSKGKDPLVSAEWDSRGAPYVRCTYDADLIDTQAQVDQFVSMFGESPSL
AERYCMRGVKNTAGELVSRVSSDADPAGGWCRKWYSAHRGPDQDAALGSFCIKNPGAADC
"""

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
    conn.close()


def insert_sequences(db_file, fasta_file):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    with open(fasta_file, 'r') as file:
        current_id = None
        current_sequence = ''
        for line in file:
            if line.startswith('>'):
                if current_id is not None:
                    cursor.execute('INSERT OR REPLACE INTO sequences VALUES (?, ?)', (current_id, current_sequence))
                current_id = line.strip().split()[0].split('UniRef100_')[1]  # Extract the UniRef ID
                current_sequence = ''
            else:
                current_sequence += line.strip()
        if current_id is not None:
            cursor.execute('INSERT OR REPLACE INTO sequences VALUES (?, ?)', (current_id, current_sequence))

    conn.commit()
    conn.close()


def get_sequence(db_file, uniref_id):
    start_time = time.time()
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    db_load_time = time.time() - start_time
    print(f"DB load time: {db_load_time:.2f} seconds")
    cursor.execute('SELECT sequence FROM sequences WHERE uniref_id = ?', (uniref_id,))
    result = cursor.fetchone()

    conn.close()

    if result:
        return result[0]
    else:
        return None

def test_retrieval_speed(db_file):
    uniref_ids = ['Q6GZX4', 'Q6GZX3', 'A0A1G9Q4Q5', 'A0A915H853', 'A0A1N7ASK9',
                  'A0A5N3UL60', 'UPI00085FBCE5',
                  'UPI00085C2E44', 'UPI00085C060D', 'UPI00085BD2BC',
                  ]
    start_time = time.time()
    for uniref_id in uniref_ids:
        sequence = get_sequence(db_file, uniref_id)
        if sequence:
            print(f"Sequence for {uniref_id}:")
            print(sequence)
        else:
            print(f"Sequence not found for {uniref_id}")
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds for {len(uniref_ids)} sequences")


if __name__ == '__main__':
    # Create the database
    db_file = '/SAN/orengolab/cath_plm/ProFam/data/uniref100/uniref100_sequences.db'
    create_database(db_file)
    # Insert sequences into the database
    fasta_file = '/SAN/orengolab/cath_plm/ProFam/data/uniref100/uniref100.fasta'
    insert_sequences(db_file, fasta_file)

    # Example usage
    uniref_id = 'Q6GZX4'
    sequence = get_sequence(db_file, uniref_id)
    if sequence:
        print(f"Sequence for {uniref_id}:")
        print(sequence)
    else:
        print(f"Sequence not found for {uniref_id}")

    test_retrieval_speed(db_file)
