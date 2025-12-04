import sqlite3
import os
import json
import hashlib
from threading import Lock
from typing import Optional, Any, Dict, List


def _get_db_path():
    data_dir = os.path.join(os.getcwd(), "data")
    if not os.path.exists(data_dir):
        try:
            os.makedirs(data_dir, exist_ok=True)
        except Exception:
            import tempfile
            data_dir = tempfile.gettempdir()
    return os.path.join(data_dir, 'molmo-r1-gemini-cache.sqlite3')


DB_PATH = _get_db_path()
_db_lock = Lock()


def _init_db():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                '''CREATE TABLE IF NOT EXISTS gemini_cache (
                    cache_key TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    model TEXT NOT NULL,
                    temperature REAL NOT NULL,
                    media_paths TEXT,
                    media_type TEXT,
                    use_vertexai INTEGER NOT NULL,
                    response TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )'''
            )
            conn.execute(
                '''CREATE INDEX IF NOT EXISTS idx_gemini_cache_key ON gemini_cache(cache_key)'''
            )
    except Exception as e:
        print(f"Error initializing Gemini cache database at {DB_PATH}: {e}")
        raise


try:
    _init_db()
except Exception as e:
    print(f"Error initializing Gemini cache database: {e}")


def _generate_cache_key(query: str, model: str, temperature: float, media_paths: Optional[List[str]], media_type: Optional[str], use_vertexai: bool) -> str:
    key_data = {
        "query": query,
        "model": model,
        "temperature": temperature,
        "media_paths": media_paths or [],
        "media_type": media_type or None,
        "use_vertexai": bool(use_vertexai),
    }
    key_string = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_string.encode('utf-8')).hexdigest()


def get_cached_response(query: str, model: str, temperature: float, media_paths: Optional[List[str]] = None, media_type: Optional[str] = None, use_vertexai: bool = False) -> Optional[Dict[str, Any]]:
    cache_key = _generate_cache_key(query, model, temperature, media_paths, media_type, use_vertexai)
    try:
        with _db_lock, sqlite3.connect(DB_PATH) as conn:
            cur = conn.execute(
                'SELECT response FROM gemini_cache WHERE cache_key=?',
                (cache_key,)
            )
            row = cur.fetchone()
            if row:
                try:
                    cached_data = json.loads(row[0])
                    return cached_data
                except Exception as e:
                    print(f"Error loading cached Gemini response: {e}")
                    return None
            return None
    except Exception as e:
        print(f"Error accessing Gemini cache database: {e}")
        return None


def set_cached_response(query: str, model: str, temperature: float, response: Dict[str, Any], media_paths: Optional[List[str]] = None, media_type: Optional[str] = None, use_vertexai: bool = False):
    cache_key = _generate_cache_key(query, model, temperature, media_paths, media_type, use_vertexai)
    try:
        with _db_lock, sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                '''INSERT OR REPLACE INTO gemini_cache 
                   (cache_key, query, model, temperature, media_paths, media_type, use_vertexai, response) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    cache_key,
                    query,
                    model,
                    temperature,
                    json.dumps(media_paths) if media_paths else None,
                    media_type,
                    1 if use_vertexai else 0,
                    json.dumps(response)
                )
            )
            conn.commit()
    except Exception as e:
        print(f"Error storing Gemini response in cache: {e}")


def clear_cache():
    try:
        with _db_lock, sqlite3.connect(DB_PATH) as conn:
            conn.execute('DELETE FROM gemini_cache')
            conn.commit()
    except Exception as e:
        print(f"Error clearing Gemini cache: {e}")


def get_cache_stats() -> Dict[str, int]:
    try:
        with _db_lock, sqlite3.connect(DB_PATH) as conn:
            cur = conn.execute('SELECT COUNT(*) FROM gemini_cache')
            total_entries = cur.fetchone()[0]
            cur = conn.execute('SELECT COUNT(DISTINCT model) FROM gemini_cache')
            unique_models = cur.fetchone()[0]
            return {
                'total_entries': total_entries,
                'unique_models': unique_models
            }
    except Exception as e:
        print(f"Error getting Gemini cache stats: {e}")
        return {
            'total_entries': 0,
            'unique_models': 0
        }


