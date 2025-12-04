"""
Cache utilities for timestamp identification and reasoning operations.
"""
import sqlite3
import json
import hashlib
import threading
from typing import Dict, Any, Optional, List
import os

def _get_db_path():
    data_dir = os.path.join(os.getcwd(), "data")
    if not os.path.exists(data_dir):
        try:
            os.makedirs(data_dir, exist_ok=True)
        except Exception:
            import tempfile
            data_dir = tempfile.gettempdir()
    return os.path.join(data_dir, 'timestamp_reasoning_cache.db')


# Database path for caching
DB_PATH = _get_db_path()
_db_lock = threading.Lock()

def _init_cache_db():
    """Initialize the cache database with required tables."""
    try:
        with _db_lock, sqlite3.connect(DB_PATH) as conn:
            # Create timestamp identification cache table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS timestamp_cache (
                    cache_key TEXT PRIMARY KEY,
                    video_path TEXT,
                    event TEXT,
                    timestamp_start TEXT,
                    timestamp_end TEXT,
                    response TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create reasoning cache table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS reasoning_cache (
                    cache_key TEXT PRIMARY KEY,
                    query TEXT,
                    media_paths TEXT,
                    media_type TEXT,
                    response TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for better performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp_video_path ON timestamp_cache(video_path)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp_event ON timestamp_cache(event)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_reasoning_query ON reasoning_cache(query)')
            
            conn.commit()
    except Exception as e:
        print(f"Error initializing cache database: {e}")

def _generate_timestamp_cache_key(video_path: str, event: str, timestamp_start: str, timestamp_end: str, model_name: str = None) -> str:
    """Generate cache key for timestamp identification."""
    key_data = {
        'video_path': video_path,
        'event': event,
        'timestamp_start': timestamp_start,
        'timestamp_end': timestamp_end,
        'model_name': model_name or 'default'
    }
    key_string = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_string.encode('utf-8')).hexdigest()

def _generate_reasoning_cache_key(query: str, media_paths: List[str], media_type: str, model_name: str = None) -> str:
    """Generate cache key for reasoning operations."""
    key_data = {
        'query': query,
        'media_paths': sorted(media_paths) if media_paths else [],
        'media_type': media_type,
        'model_name': model_name or 'default'
    }
    key_string = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_string.encode('utf-8')).hexdigest()

def get_cached_timestamp_response(video_path: str, event: str, timestamp_start: str, timestamp_end: str, model_name: str = None) -> Optional[Dict[str, Any]]:
    """Get cached timestamp identification response."""
    cache_key = _generate_timestamp_cache_key(video_path, event, timestamp_start, timestamp_end, model_name)
    
    try:
        with _db_lock, sqlite3.connect(DB_PATH) as conn:
            cur = conn.execute(
                'SELECT response FROM timestamp_cache WHERE cache_key=?',
                (cache_key,)
            )
            row = cur.fetchone()
            if row:
                try:
                    cached_data = json.loads(row[0])
                    return cached_data
                except Exception as e:
                    print(f"Error loading cached timestamp response: {e}")
                    return None
            return None
    except Exception as e:
        print(f"Error accessing timestamp cache database: {e}")
        return None

def set_cached_timestamp_response(video_path: str, event: str, timestamp_start: str, timestamp_end: str, response: Dict[str, Any], model_name: str = None):
    """Cache timestamp identification response."""
    cache_key = _generate_timestamp_cache_key(video_path, event, timestamp_start, timestamp_end, model_name)
    
    try:
        with _db_lock, sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                'INSERT OR REPLACE INTO timestamp_cache (cache_key, video_path, event, timestamp_start, timestamp_end, response) VALUES (?, ?, ?, ?, ?, ?)',
                (cache_key, video_path, event, timestamp_start, timestamp_end, json.dumps(response))
            )
            conn.commit()
    except Exception as e:
        print(f"Error caching timestamp response: {e}")

def get_cached_reasoning_response(query: str, media_paths: List[str], media_type: str, model_name: str = None) -> Optional[Dict[str, Any]]:
    """Get cached reasoning response."""
    cache_key = _generate_reasoning_cache_key(query, media_paths, media_type, model_name)
    
    try:
        with _db_lock, sqlite3.connect(DB_PATH) as conn:
            cur = conn.execute(
                'SELECT response FROM reasoning_cache WHERE cache_key=?',
                (cache_key,)
            )
            row = cur.fetchone()
            if row:
                try:
                    cached_data = json.loads(row[0])
                    return cached_data
                except Exception as e:
                    print(f"Error loading cached reasoning response: {e}")
                    return None
            return None
    except Exception as e:
        print(f"Error accessing reasoning cache database: {e}")
        return None

def set_cached_reasoning_response(query: str, media_paths: List[str], media_type: str, response: Dict[str, Any], model_name: str = None):
    """Cache reasoning response."""
    cache_key = _generate_reasoning_cache_key(query, media_paths, media_type, model_name)
    
    try:
        with _db_lock, sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                'INSERT OR REPLACE INTO reasoning_cache (cache_key, query, media_paths, media_type, response) VALUES (?, ?, ?, ?, ?)',
                (cache_key, query, json.dumps(media_paths), media_type, json.dumps(response))
            )
            conn.commit()
    except Exception as e:
        print(f"Error caching reasoning response: {e}")

def clear_cache(cache_type: str = "all"):
    """Clear cache entries."""
    try:
        with _db_lock, sqlite3.connect(DB_PATH) as conn:
            if cache_type == "all":
                conn.execute('DELETE FROM timestamp_cache')
                conn.execute('DELETE FROM reasoning_cache')
            elif cache_type == "timestamp":
                conn.execute('DELETE FROM timestamp_cache')
            elif cache_type == "reasoning":
                conn.execute('DELETE FROM reasoning_cache')
            conn.commit()
            print(f"Cleared {cache_type} cache")
    except Exception as e:
        print(f"Error clearing cache: {e}")

def get_cache_stats() -> Dict[str, int]:
    """Get cache statistics."""
    try:
        with _db_lock, sqlite3.connect(DB_PATH) as conn:
            timestamp_count = conn.execute('SELECT COUNT(*) FROM timestamp_cache').fetchone()[0]
            reasoning_count = conn.execute('SELECT COUNT(*) FROM reasoning_cache').fetchone()[0]
            return {
                'timestamp_cache_entries': timestamp_count,
                'reasoning_cache_entries': reasoning_count,
                'total_entries': timestamp_count + reasoning_count
            }
    except Exception as e:
        print(f"Error getting cache stats: {e}")
        return {'timestamp_cache_entries': 0, 'reasoning_cache_entries': 0, 'total_entries': 0}

# Initialize the cache database on import
_init_cache_db()
