import sqlite3
import os
import json
from threading import Lock

DB_PATH = os.path.join("data", 'sage-search-cache.sqlite3')
_db_lock = Lock()

# Ensure the table exists
def _init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            '''CREATE TABLE IF NOT EXISTS cache (
                search_type TEXT NOT NULL,
                query TEXT NOT NULL,
                result TEXT,
                PRIMARY KEY (search_type, query)
            )'''
        )

try:
    _init_db()
except Exception as e:
    print(f"Error initializing database: {e}")

def get_cached_result(query: str, search_type: str):
    """Return cached result for (query, search_type) if present, else None."""
    with _db_lock, sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            'SELECT result FROM cache WHERE search_type=? AND query=?',
            (search_type, query)
        )
        row = cur.fetchone()
        if row:
            try:
                return json.loads(row[0])
            except Exception:
                return None
        return None

def set_cached_result(query: str, search_type: str, result):
    """Store result for (query, search_type)."""
    with _db_lock, sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            'INSERT OR REPLACE INTO cache (search_type, query, result) VALUES (?, ?, ?)',
            (search_type, query, json.dumps(result))
        )
        conn.commit() 