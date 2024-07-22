import sqlite3

# Connect to the database
conn = sqlite3.connect('/SAN/orengolab/cath_plm/profam_db/profam.db')
cursor = conn.cursor()

# Get table names
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

print("Tables in the database:")
for table in tables:
    print(table[0])
    # Get column information for each table
    cursor.execute(f"PRAGMA table_info({table[0]})")
    columns = cursor.fetchall()
    print("Columns:")
    for column in columns:
        print(f"  {column[1]} ({column[2]})")
    print()

# Execute a query to count the rows in the sequences table
cursor.execute("SELECT COUNT(*) FROM sequences")

# Fetch the result
row_count = cursor.fetchone()[0]

print(f"Number of rows in the sequences table: {row_count}")