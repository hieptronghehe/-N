import sqlite3

# Make temp table if doesn't exist
def create_temp_table(db_path):
    # create table with correct INTEGER keyword and use context manager
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS temp_data
            (x_location INTEGER, y_location INTEGER, person_ID INTEGER)
        """)
        conn.commit()


# Query the database and return all records
def show_all(db_path):
    # use context manager; no commit needed for SELECT
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM temp_data")
        for item in cur:
            print(item)


# Add many data to the table
def add_many_temp(db_path, data_list):
    """
    Insert multiple (x_location, y_location, person_ID) records into temp_data.
    data_list should be an iterable of (x, y) tuples.
    """
    if not data_list:
        return

    # enforce that each item is a 3-tuple (x, y, person_id)
    normalized = []
    for i, item in enumerate(data_list):
        if not (isinstance(item, (list, tuple)) and len(item) == 3):
            raise ValueError(f"data_list item at index {i} must be a tuple/list of length 3 (x,y,person_id). Got: {item}")
        normalized.append(tuple(item))

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.executemany("INSERT INTO temp_data (x_location, y_location, person_ID) VALUES (?, ?, ?)", normalized)
        conn.commit()

# Delete all data in temp_file
def delete_temp(db_path):
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM temp_data")
        conn.commit()