import sqlite3
import os
import json
import hashlib
from threading import Lock
from typing import Optional, Any, Dict, List

# Create a more robust database path
def _get_db_path():
    # Try to use a data directory in the current working directory first
    data_dir = os.path.join(os.getcwd(), "data")
    if not os.path.exists(data_dir):
        # If data directory doesn't exist, try to create it
        try:
            os.makedirs(data_dir, exist_ok=True)
        except Exception:
            # If we can't create the data directory, fall back to temp directory
            import tempfile
            data_dir = tempfile.gettempdir()
    
    return os.path.join(data_dir, 'molmo-r1-gpt-cache.sqlite3')

DB_PATH = _get_db_path()
_db_lock = Lock()

# Ensure the table exists
def _init_db():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                '''CREATE TABLE IF NOT EXISTS gpt_cache (
                    cache_key TEXT PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    model TEXT NOT NULL,
                    temperature REAL NOT NULL,
                    image_urls TEXT,
                    history TEXT,
                    response TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )'''
            )
            # Create index for faster lookups
            conn.execute(
                '''CREATE INDEX IF NOT EXISTS idx_gpt_cache_key ON gpt_cache(cache_key)'''
            )
    except Exception as e:
        print(f"Error initializing GPT cache database at {DB_PATH}: {e}")
        raise

try:
    _init_db()
except Exception as e:
    print(f"Error initializing GPT cache database: {e}")

def _generate_cache_key(prompt: str, model: str, temperature: float, image_urls: Optional[List[str]] = None, history: Optional[List[Dict]] = None) -> str:
    """Generate a unique cache key for GPT API requests."""
    # Create a hash of all the parameters that affect the response
    key_data = {
        "prompt": prompt,
        "model": model,
        "temperature": temperature,
        "image_urls": image_urls or [],
        "history": history or []
    }
    
    # Convert to JSON string and hash it
    key_string = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_string.encode('utf-8')).hexdigest()

def get_cached_response(prompt: str, model: str, temperature: float, image_urls: Optional[List[str]] = None, history: Optional[List[Dict]] = None) -> Optional[tuple]:
    """Return cached response for GPT request if present, else None.
    
    Returns:
        tuple: (response_content, return_messages) if cached, else None
    """
    cache_key = _generate_cache_key(prompt, model, temperature, image_urls, history)
    
    try:
        with _db_lock, sqlite3.connect(DB_PATH) as conn:
            cur = conn.execute(
                'SELECT response FROM gpt_cache WHERE cache_key=?',
                (cache_key,)
            )
            row = cur.fetchone()
            if row:
                try:
                    cached_data = json.loads(row[0])
                    return cached_data['response_content'], cached_data['return_messages']
                except Exception as e:
                    print(f"Error loading cached GPT response: {e}")
                    return None
            return None
    except Exception as e:
        print(f"Error accessing GPT cache database: {e}")
        return None

def set_cached_response(prompt: str, model: str, temperature: float, response_content: str, return_messages: List[Dict], image_urls: Optional[List[str]] = None, history: Optional[List[Dict]] = None):
    """Store GPT response in cache."""
    cache_key = _generate_cache_key(prompt, model, temperature, image_urls, history)
    
    cached_data = {
        'response_content': response_content,
        'return_messages': return_messages
    }
    
    try:
        with _db_lock, sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                '''INSERT OR REPLACE INTO gpt_cache 
                   (cache_key, prompt, model, temperature, image_urls, history, response) 
                   VALUES (?, ?, ?, ?, ?, ?, ?)''',
                (
                    cache_key,
                    prompt,
                    model,
                    temperature,
                    json.dumps(image_urls) if image_urls else None,
                    json.dumps(history) if history else None,
                    json.dumps(cached_data)
                )
            )
            conn.commit()
    except Exception as e:
        print(f"Error storing GPT response in cache: {e}")
        # Don't raise the exception to avoid breaking the main flow

def clear_cache():
    """Clear all cached GPT responses."""
    try:
        with _db_lock, sqlite3.connect(DB_PATH) as conn:
            conn.execute('DELETE FROM gpt_cache')
            conn.commit()
    except Exception as e:
        print(f"Error clearing GPT cache: {e}")

def get_cache_stats() -> Dict[str, int]:
    """Get cache statistics."""
    try:
        with _db_lock, sqlite3.connect(DB_PATH) as conn:
            cur = conn.execute('SELECT COUNT(*) FROM gpt_cache')
            total_entries = cur.fetchone()[0]
            
            cur = conn.execute('SELECT COUNT(DISTINCT model) FROM gpt_cache')
            unique_models = cur.fetchone()[0]
            
            return {
                'total_entries': total_entries,
                'unique_models': unique_models
            }
    except Exception as e:
        print(f"Error getting GPT cache stats: {e}")
        return {
            'total_entries': 0,
            'unique_models': 0
        }
