import sqlite3

# Make temp table if doesn't exist
def create_temp_table(db_path):
    # create table with correct INTEGER keyword and use context manager
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        # person_ID is primary key so we can upsert by person_ID
        cur.execute("""
            CREATE TABLE IF NOT EXISTS temp_data
            (person_ID INTEGER PRIMARY KEY, x_location INTEGER, y_location INTEGER)
        """)
        conn.commit()
        # ensure two rows exist for person_ID 1 and 2 (initialized with NULL coords)
        cur.execute("SELECT person_ID FROM temp_data WHERE person_ID IN (1,2)")
        existing = {row[0] for row in cur.fetchall()}
        to_add = []
        for pid in (1, 2):
            if pid not in existing:
                to_add.append((pid, None, None))
        if to_add:
            cur.executemany("INSERT OR IGNORE INTO temp_data (person_ID, x_location, y_location) VALUES (?, ?, ?)", to_add)
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

    # Only accept triples (x, y, person_id) and only person_id 1 or 2
    allowed_pids = {1, 2}
    filtered = []
    for x, y, pid in normalized:
        try:
            pid_int = int(pid)
        except Exception:
            continue
        if pid_int not in allowed_pids:
            # skip unknown person IDs
            continue
        # coerce x,y to integers or NULL
        try:
            x_i = int(round(float(x)))
        except Exception:
            x_i = None
        try:
            y_i = int(round(float(y)))
        except Exception:
            y_i = None
        filtered.append((pid_int, x_i, y_i))

    if not filtered:
        return

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        # upsert by person_ID: replace existing row for that person_ID
        cur.executemany(
            "INSERT OR REPLACE INTO temp_data (person_ID, x_location, y_location) VALUES (?, ?, ?)",
            filtered,
        )
        conn.commit()

# Delete all data in temp_file
def delete_temp(db_path):
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM temp_data")
        conn.commit()