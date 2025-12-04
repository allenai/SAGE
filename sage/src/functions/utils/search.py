from typing import Dict, Any
import requests
import os
from serpapi import GoogleSearch
from sage.utils.utils import WEB_SEARCH
from sage.src.functions.utils.utils import is_url, upload_to_gcp_bucket
import os
import requests
from typing import Dict, Any
import time
from sage.src.functions.utils import search_db
import hashlib
import http.client
import json


def _file_sha256(path):
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def _reverse_image_search(image_path: str, num_results: int) -> Dict[str, Any]:
    """Perform a reverse image search using SerpApi's Google Lens endpoint."""
    if not image_path:
        return {
            "error": "image_path is required for reverse image search",
            "results": [],
        }

    # Use hash of file as cache key if local file, else use URL string
    if is_url(image_path):
        cache_key = image_path
    else:
        if not os.path.exists(image_path):
            return {"error": f"Image file not found: {image_path}", "results": []}
        cache_key = _file_sha256(image_path)

    cached = search_db.get_cached_result(cache_key, "reverse_image")
    if cached:
        return cached

    serp_api_key = WEB_SEARCH.get("serp_api_key")
    gcp_bucket = WEB_SEARCH.get("gcp_bucket")

    if not serp_api_key:
        return {"error": "SerpApi key not configured.", "results": []}

    if is_url(image_path):
        image_url = image_path
    else:
        if not os.path.exists(image_path):
            return {"error": f"Image file not found: {image_path}", "results": []}
        if not gcp_bucket:
            return {"error": "GCP bucket not configured.", "results": []}

        dest_blob_name = os.path.basename(image_path)
        try:
            image_url = upload_to_gcp_bucket(image_path, gcp_bucket, dest_blob_name)
        except Exception as e:
            return {"error": f"Failed to upload to GCP: {e}", "results": []}

    params = {"engine": "google_lens", "url": image_url, "api_key": serp_api_key}

    for attempt in range(3):
        try:
            search = GoogleSearch(params)
            data = search.get_dict()

            if "visual_matches" not in data:
                return {
                    "query_image": image_url,
                    "error": "No visual matches found or unexpected response.",
                    "results": [],
                }

            results = []
            for item in data["visual_matches"][:num_results]:
                results.append(
                    {
                        "title": item.get("title", ""),
                        "link": item.get("link", ""),
                        "image_url": item.get("thumbnail", ""),
                    }
                )

            # Store in cache
            search_db.set_cached_result(cache_key, "reverse_image", {
                "search_type": "reverse_image",
                "query_image": image_url,
                "results": results,
            })
            return {
                "search_type": "reverse_image",
                "query_image": image_url,
                "results": results,
            }
        except Exception as e:
            if "429 Client Error: Too Many Requests" in str(e) and attempt < 2:
                time.sleep(60)
                continue
            return {
                "search_type": "reverse_image",
                "query_image": image_url,
                "error": str(e),
                "results": [],
            }


def _serper_api_request(endpoint: str, payload: dict, api_key: str) -> dict:
    """Helper to make a POST request to Serper API using http.client."""
    conn = http.client.HTTPSConnection("google.serper.dev")
    headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }
    conn.request("POST", endpoint, json.dumps(payload), headers)
    res = conn.getresponse()
    data = res.read()
    try:
        return json.loads(data.decode("utf-8"))
    except Exception as e:
        return {"error": f"Failed to decode Serper API response: {e}", "raw": data}


def _google_custom_search_request(endpoint: str, params: dict, result_type: str, num_results: int) -> dict:
    """Helper to make a GET request to Google Custom Search API and extract results."""
    results = []
    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        search_data = response.json()
        if "items" in search_data:
            for item in search_data["items"][:num_results]:
                if result_type == "web":
                    results.append({
                        "title": item.get("title", ""),
                        "snippet": item.get("snippet", ""),
                        "link": item.get("link", ""),
                    })
                elif result_type == "image":
                    results.append({
                        "title": item.get("title", ""),
                        "image_url": item.get("link", ""),
                        "thumbnail": item.get("image", {}).get("thumbnailLink", ""),
                        "context_url": item.get("image", {}).get("contextLink", ""),
                    })
        return {"results": results, "error": None}
    except requests.exceptions.RequestException as e:
        return {"results": [], "error": str(e)}


def _web_search(query: str, num_results: int, engine: str = "serper") -> Dict[str, Any]:
    """Perform a web search using Google Custom Search API or Serper API."""
    if not query:
        return {"error": "query is required for web search", "results": []}

    cache_key = query
    cached = search_db.get_cached_result(cache_key, "web")
    if cached:
        return cached

    results = []
    if engine == "serper":
        serper_api_key = WEB_SEARCH.get("serper_api_key")
        if not serper_api_key:
            return {"error": "Serper API key not configured.", "results": []}
        payload = {"q": query, "num": min(num_results, 10)}
        for attempt in range(3):
            try:
                search_data = _serper_api_request("/search", payload, serper_api_key)
                if "error" in search_data:
                    return {"search_type": "web", "query": query, "error": search_data["error"], "results": []}
                if "organic" in search_data:
                    for item in search_data["organic"][:num_results]:
                        results.append({
                            "title": item.get("title", ""),
                            "snippet": item.get("snippet", ""),
                            "link": item.get("link", ""),
                        })
                search_db.set_cached_result(cache_key, "web", {"search_type": "web", "query": query, "results": results})
                return {"search_type": "web", "query": query, "results": results}
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    time.sleep(60)
                    continue
                return {"search_type": "web", "query": query, "error": str(e), "results": []}
    else:
        endpoint = "https://www.googleapis.com/customsearch/v1"
        params = {
            "q": query,
            "key": WEB_SEARCH.get("api_key"),
            "cx": WEB_SEARCH.get("cx"),
            "num": min(num_results, 10),
            "fields": "items(title,snippet,link)",
        }
        for attempt in range(3):
            result = _google_custom_search_request(endpoint, params, "web", num_results)
            if result["error"] is None:
                results = result["results"]
                search_db.set_cached_result(cache_key, "web", {"search_type": "web", "query": query, "results": results})
                return {"search_type": "web", "query": query, "results": results}
            elif "429 Client Error: Too Many Requests" in str(result["error"]) and attempt < 2:
                time.sleep(60)
                continue
            else:
                return {"search_type": "web", "query": query, "error": result["error"], "results": []}


def _image_search(query: str, num_results: int, engine: str = "serper") -> Dict[str, Any]:
    """Perform an image search using Google Custom Search API or Serper API."""
    if not query:
        return {"error": "query is required for image search", "results": []}

    cache_key = query
    cached = search_db.get_cached_result(cache_key, "image")
    if cached:
        return cached

    results = []
    if engine == "serper":
        serper_api_key = WEB_SEARCH.get("serper_api_key")
        if not serper_api_key:
            return {"error": "Serper API key not configured.", "results": []}
        payload = {"q": query, "num": min(num_results, 10)}
        for attempt in range(3):
            try:
                search_data = _serper_api_request("/images", payload, serper_api_key)
                if "error" in search_data:
                    return {"search_type": "image", "query": query, "error": search_data["error"], "results": []}
                if "images" in search_data:
                    for item in search_data["images"][:num_results]:
                        results.append({
                            "title": item.get("title", ""),
                            "image_url": item.get("imageUrl", ""),
                            "thumbnail": item.get("thumbnailUrl", ""),
                            "context_url": item.get("pageUrl", ""),
                        })
                search_db.set_cached_result(cache_key, "image", {"search_type": "image", "query": query, "results": results})
                return {"search_type": "image", "query": query, "results": results}
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    time.sleep(60)
                    continue
                return {"search_type": "image", "query": query, "error": str(e), "results": []}
    else:
        endpoint = "https://www.googleapis.com/customsearch/v1"
        params = {
            "q": query,
            "key": WEB_SEARCH.get("api_key"),
            "cx": WEB_SEARCH.get("cx"),
            "num": min(num_results, 10),
            "searchType": "image",
            "fields": "items(title,link,image)",
        }
        for attempt in range(3):
            result = _google_custom_search_request(endpoint, params, "image", num_results)
            if result["error"] is None:
                results = result["results"]
                search_db.set_cached_result(cache_key, "image", {"search_type": "image", "query": query, "results": results})
                return {"search_type": "image", "query": query, "results": results}
            elif "429 Client Error: Too Many Requests" in str(result["error"]) and attempt < 2:
                time.sleep(60)
                continue
            else:
                return {"search_type": "image", "query": query, "error": result["error"], "results": []}