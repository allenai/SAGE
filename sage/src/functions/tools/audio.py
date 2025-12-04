from typing import Optional, List, Dict, Any, Tuple
from sage.src.functions.utils.transcribe import transcribe_video
from sage.src.functions.utils.temporal import get_video_duration, fix_timestamp
import os

def verbal_transcript(
    video_path: str, 
    timestamp_start: str, 
    timestamp_end: str,
) -> Dict[str, Any]:
    """
    Perform automatic speech recognition on the video's verbal information using WhisperX.
    This function provides enhanced transcription with segment-level timestamps.
    You MUST keep the difference between the start and end timestamps below 10 minutes.

    Args:
        video_path: Path to the video file
        timestamp_start: Start timestamp (HH:MM:SS)
        timestamp_end: End timestamp (HH:MM:SS)
    Returns:
        Dictionary containing the segment-level verbal transcript of the video
    """
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file does not exist: {video_path}")
    
    video_duration = get_video_duration(video_path)
    if timestamp_start is not None:
        timestamp_start = fix_timestamp(timestamp_start, video_duration)
    if timestamp_end is not None:
        timestamp_end = fix_timestamp(timestamp_end, video_duration)
    transcript = transcribe_video(
        video_path,
        timestamp_start=timestamp_start,
        timestamp_end=timestamp_end,
    )
    if len(transcript) == 0:
        return "Hmm, we might have been given a video with no verbal speech, so there is nothing to transcribe."
    return {"transcript": str(transcript)}
