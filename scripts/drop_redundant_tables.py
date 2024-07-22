import sqlite3

# Connect to the database
conn = sqlite3.connect('/SAN/orengolab/cath_plm/profam_db/profam.db')
cursor = conn.cursor()

# SQL statements to drop the tables
drop_families = "DROP TABLE IF EXISTS families;"
drop_protein_families = "DROP TABLE IF EXISTS protein_families;"

try:
    # Execute the DROP TABLE statements
    cursor.execute(drop_families)
    print("Table 'families' dropped successfully.")

    cursor.execute(drop_protein_families)
    print("Table 'protein_families' dropped successfully.")

    # Commit the changes
    conn.commit()

except sqlite3.Error as e:
    print(f"An error occurred: {e}")

# Close the connection
conn.close()