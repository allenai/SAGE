import base64
from typing import Dict, Any
import requests
import os
from pathlib import Path
import json
from sage.utils.utils import WEB_SEARCH
from sage.src.functions.utils.search import (
    _web_search,
)
from sage.src.functions.utils.parse_web import parse_website
from sage.src.functions.utils.search_db import get_cached_result, set_cached_result


def unified_web_search(
    query: str,
    search_type: str = "web",
    num_results: int = 3,
) -> Dict[str, Any]:
    """
    Unified function to perform web search or image search using a text query.

    Args:
        query: Search query (required for web and image search)
        search_type: Type of search - "web" or  "image"
        num_results: Number of search results to return (max 10)

    Returns:
        Dictionary containing search results
    """

    if num_results is None:
        num_results = 3

    return _web_search(query, num_results)


def parse_web_data(website_url: str, max_content_length: int = 5000) -> Dict[str, Any]:
    """
    Parse web data from a given URL. 
    We only parse a webpage's content if available title and snippet 
    does not have enough information to answer the query.

    Args:
       website_url: The URL of the website to parse.
       max_content_length: The maximum length of the content to parse. Tune this depending on where the reqd info may be on the webpage based on the query.

    Returns:
       A dictionary containing the text content of the webpage.
    """
    # 1. Try cache first
    cached = get_cached_result(website_url, search_type="web_parse")
    if cached:
        # Truncate content if needed
        if "content" in cached and len(cached["content"]) > max_content_length:
            cached["content"] = cached["content"][:max_content_length]
        return cached

    try:
        parsed_content = parse_website(website_url, max_content_length=5000)
        result = {
            "title": parsed_content.title,
            "content": parsed_content.main_content,
            "url": parsed_content.url,
        }
        set_cached_result(website_url, "web_parse", result)
        return result
    except Exception as e:
        try:
            api_key = WEB_SEARCH.get('serper_api_key')
            if not api_key:
                raise ValueError("Serper API key not found")
                
            endpoint = "https://scrape.serper.dev/"
            headers = {
                'X-API-KEY': api_key,
                'Content-Type': 'application/json'
            }
            payload = json.dumps({"url": website_url})
            response = requests.post(endpoint, headers=headers, data=payload, timeout=30)
            response.raise_for_status()
            data = response.json()

            title = data.get("title", "")
            content = data.get("text", "")
            if len(content) > max_content_length:
                content = content[:max_content_length]
            url = data.get("url", website_url)

            result = {
                "title": title,
                "content": content,
                "url": url,
            }
            set_cached_result(website_url, "web_parse", result)
            return result
            
        except requests.exceptions.RequestException as req_err:
            # Handle network/HTTP errors
            return {
                "error": f"Network error during web scraping: {str(req_err)}",
                "title": "",
                "content": "",
                "url": website_url,
            }
        except json.JSONDecodeError as json_err:
            # Handle JSON parsing errors
            return {
                "error": f"Invalid JSON response from scraping API: {str(json_err)}",
                "title": "",
                "content": "",
                "url": website_url,
            }
        except ValueError as val_err:
            # Handle missing API key or other value errors
            return {
                "error": f"Configuration error: {str(val_err)}",
                "title": "",
                "content": "",
                "url": website_url,
            }
        except Exception as fallback_err:
            # Handle any other unexpected errors
            return {
                "error": f"Unexpected error during web scraping: {str(fallback_err)}",
                "title": "",
                "content": "",
                "url": website_url,
            }


if __name__ == "__main__":
    from icecream import ic

    ic(unified_web_search(query="the owner of HAAS", search_type="web", num_results=5))
    # ic(parse_web_data("https://en.wikipedia.org/wiki/Paris"))
