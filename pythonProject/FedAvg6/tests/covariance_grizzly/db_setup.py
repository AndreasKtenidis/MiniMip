import duckdb
import os

csv_path = os.path.join(os.path.dirname(__file__), '../../data/california_housing/housing.csv')
db_path = os.path.join(os.path.dirname(__file__), '../../data/california_housing/housing.duckdb')
con = duckdb.connect(database=db_path, read_only=False)

# Create table from CSV
con.execute(f"CREATE TABLE IF NOT EXISTS housing AS SELECT * FROM read_csv_auto('{csv_path}')")
print(con.execute("DESCRIBE housing").fetchall())
con.close()
