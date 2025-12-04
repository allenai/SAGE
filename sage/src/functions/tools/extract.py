from typing import List, Dict
from sage.src.functions.utils.extract import extract_frames, extract_subclip
from sage.src.functions.utils.temporal import timestamp_to_seconds, get_video_duration, fix_timestamp, seconds_to_timestamp
import os

def extract_parts_from_timestamp(
    video_path: str,
    timestamp_start: str,
    timestamp_end: str,
    extract_type: str = "frames",
) -> Dict[str, List[str]]:
    """
    Extract frames or video subclips from a video between two timestamps. 
    You MUST keep the difference between the start and end timestamps below 5 minutes.

    Args:
        video_path: Path to the video file
        timestamp_start: Start timestamp (HH:MM:SS)
        timestamp_end: End timestamp (HH:MM:SS)
        extract_type: must be "frames" or "subclips"
    Returns:
        Lists of saved extracted parts.
    """

    if extract_type not in ["frames", "subclips"]:
        raise ValueError(f"Invalid extract_type: {extract_type}. Must be 'frames' or 'subclips'")
    
    if not video_path.endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
        raise ValueError(f"Video file does not have a valid extension: {video_path}")
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file does not exist: {video_path}")
        
    video_duration = get_video_duration(video_path)
    if timestamp_start is not None:
        timestamp_start = fix_timestamp(timestamp_start, video_duration)
    else:
        timestamp_start = "00:00:00"
        
    if timestamp_end is not None:
        timestamp_end = fix_timestamp(timestamp_end, video_duration)
    else:
        timestamp_end = seconds_to_timestamp(min(video_duration, 600), in_hr=True)
    
    timestamp_start_seconds = timestamp_to_seconds(timestamp_start)
    timestamp_end_seconds = timestamp_to_seconds(timestamp_end)
    if extract_type == "frames":
        paths = extract_frames(video_path, timestamp_start_seconds, timestamp_end_seconds)
        return {"media_paths": paths}
    else:  # subclips
        path = extract_subclip(video_path, timestamp_start_seconds, timestamp_end_seconds)
        return {"media_paths": [path]}
