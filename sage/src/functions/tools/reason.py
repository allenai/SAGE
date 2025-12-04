from typing import List, Dict, Any
from sage.src.api.response import get_response
import os

USE_GPT_AS_TOOL = os.getenv("USE_GPT_AS_TOOL", "False").lower() == "true"

def perform_reasoning(query: str, media_paths: List[str]) -> Dict[str, Any]:
    """
    Reason over a given query with the help of the given images (media_paths). 
    The inputs can also just be text, this is an LLM model call.

    Args:
        query: Query to reason over
        media_paths: List of media paths (frame paths)
    Returns:
        Dictionary containing the answer to the query.
    """
    if media_paths is not None and len(media_paths) > 0:
        for media_path in media_paths:
            if not os.path.exists(media_path):
                raise FileNotFoundError(f"Media file does not exist: {media_path}, in the passed media list: {media_paths}")

    if len(media_paths) > 0:
        is_video = any(media_path.endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")) for media_path in media_paths)
        if is_video:
            media_type = "video"
        else:
            media_type = "image"
    else:
        media_type = None
    answer = get_response(
        query, 
        media_urls=media_paths, 
        media_type=media_type, 
        temperature=0.0, 
        model_name="gemini:gemini-2.5-flash" if not USE_GPT_AS_TOOL else "gpt:gpt-4o"
    )
    return {"answer": answer}
