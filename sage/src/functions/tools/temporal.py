from typing import Dict, Any
from sage.utils.utils import (
    VISUAL_TEMPORAL_GROUNDING_PROMPT,
)
from sage.src.api.response import get_response
from sage.src.functions.utils.extract import extract_subclip
from sage.src.functions.utils.temporal import timestamp_to_seconds, get_video_duration, fix_timestamp, seconds_to_timestamp
import os

USE_GPT_AS_TOOL = os.getenv("USE_GPT_AS_TOOL", "False").lower() == "true"

def identify_timestamps_visually(
    video_path: str, event: str, timestamp_start: str, timestamp_end: str
) -> Dict[str, Any]:
    """
    Identify timestamps for an event in the video.
    You MUST keep the difference between the start and end timestamps below 10 minutes. 
    You CANNOT search through the entire video.
    Use any other tool you think useful to make the best guess for the timestamps which are required arguments.

    Args:
        video_path: Path to the video file
        event: Event to locate in the video
        timestamp_start: Guessed start timestamp of the event in the video in the format of HH:MM:SS
        timestamp_end: Guessed end timestamp of the event in the video in the format of HH:MM:SS
    Returns:
        Dictionary mapping event to (start_time, end_time)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    video_duration = get_video_duration(video_path)
    timestamp_start = fix_timestamp(timestamp_start, video_duration, in_hr=True)
    timestamp_end = fix_timestamp(timestamp_end, video_duration, in_hr=True)

    prompt = (
        VISUAL_TEMPORAL_GROUNDING_PROMPT.replace("<<<event>>>", str(event))
        .replace("<<<begin>>>", timestamp_start)
        .replace("<<<end>>>", timestamp_end)
    )
    timestamp_start_seconds = timestamp_to_seconds(timestamp_start)
    timestamp_end_seconds = timestamp_to_seconds(timestamp_end)

    if timestamp_start == timestamp_end:
        raise ValueError(f"Invalid timestamps: timestamp_start {timestamp_start} and timestamp_end {timestamp_end} are the same")
    elif timestamp_start_seconds > timestamp_end_seconds:
        raise ValueError(f"Invalid timestamps: timestamp_start {timestamp_start} is greater than timestamp_end {timestamp_end}")
    elif timestamp_end_seconds > video_duration:
        raise ValueError(f"Invalid timestamps: timestamp_end {timestamp_end} is greater than the video duration {video_duration}")

    video_path = extract_subclip(video_path, timestamp_start_seconds, timestamp_end_seconds)
    response = get_response(
        prompt,
        model_name="gemini:gemini-2.5-flash" if not USE_GPT_AS_TOOL else "gpt:gpt-4o",
        media_urls=[video_path],
        media_type="video",
        temperature=0.0,
    )[0]

    if isinstance(response, dict):
        timestamps = response.get("timestamps", {})
        if timestamps.get("start", None) is not None and timestamps.get("end", None) is not None and timestamps.get("start") != timestamps.get("end"):
            if timestamp_to_seconds(timestamps["start"]) < timestamp_start_seconds:
                timestamps["start"] = seconds_to_timestamp(timestamp_start_seconds + timestamp_to_seconds(timestamps["start"]))
                timestamps["end"] = seconds_to_timestamp(timestamp_start_seconds + timestamp_to_seconds(timestamps["end"]))
        response["timestamps"] = timestamps
    return response
