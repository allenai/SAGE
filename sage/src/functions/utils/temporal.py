def timestamp_to_seconds(timestamp: str) -> float:
    """
    Convert a timestamp string to seconds.
    Args:
        timestamp (str): Timestamp string in format 'HH:MM:SS' or 'MM:SS' or decimal seconds
    Returns:
        float: Number of seconds
    """
    if timestamp.count(":") == 2:
        hour, minute, second = timestamp.split(":")
        return int(float(hour)) * 3600 + int(float(minute)) * 60 + float(second)
    elif timestamp.count(":") == 1:
        minute, second = timestamp.split(":")
        return int(float(minute)) * 60 + float(second)
    else:
        return float(timestamp)


def seconds_to_timestamp(seconds: int, in_hr: bool = False, in_mins: bool = False) -> str:
    """Convert seconds to MM:SS or HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)

    if hours > 0 or in_hr:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    elif minutes > 0 or in_mins:
        return f"{minutes:02d}:{seconds:02d}"
    else:
        return f"{seconds:02d}"

import cv2
import os

def get_video_duration(video_path: str) -> float:
    if not video_path.endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
        raise ValueError(f"Video file does not have a valid extension: {video_path}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file does not exist: {video_path}")
    cap = cv2.VideoCapture(video_path)
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
    except Exception as e:
        print(f"Failed to get video duration: {e}")
        return 0

def fix_timestamp(timestamp: str, video_duration: int, in_hr: bool = False) -> str:
    """
    Fixes timestamp overflow by considering misinterpretation of time units.
    Only applies corrections when the original timestamp significantly exceeds video duration.
    
    Args:
        timestamp (str): Original timestamp string (e.g., "01:10:00")
        video_duration (int): Duration of video in seconds
        in_hr (bool): Whether to return the timestamp in hours
    Returns:
        str: Corrected timestamp (HH:MM:SS)
    """
    from itertools import permutations
    
    def to_seconds(h, m, s):
        return h * 3600 + m * 60 + s
    
    def to_hms(seconds):
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        if h > 0 or in_hr:
            return f"{h:02}:{m:02}:{s:02}"
        else:
            return f"{m:02}:{s:02}"
    
    # Parse the timestamp parts
    parts = [int(p) for p in timestamp.split(":")]
    
    # Calculate original timestamp
    if len(parts) == 2:
        original_seconds = to_seconds(0, parts[0], parts[1])  # Assume MM:SS
    elif len(parts) == 3:
        original_seconds = to_seconds(parts[0], parts[1], parts[2])
    else:
        # Pad with zeros on the left
        padded = [0] * (3 - len(parts)) + parts
        original_seconds = to_seconds(padded[0], padded[1], padded[2])
    
    # If original timestamp is valid or only slightly over, don't reinterpret
    if original_seconds <= video_duration * 1.1:  # Allow 10% buffer
        return to_hms(min(original_seconds, video_duration))
    
    # Only try reinterpretation if timestamp is significantly over duration
    candidates = [original_seconds]
    
    if len(parts) == 2:
        # MM:SS could be HH:MM
        candidates.append(to_seconds(parts[0], parts[1], 0))
    elif len(parts) == 3:
        # Try some common misinterpretations, not all permutations
        h, m, s = parts
        if m < 60 and s < 60:
            # Maybe HH and MM were swapped
            candidates.append(to_seconds(m, h, s))
            # Maybe it's really MM:SS with leading zero hour
            candidates.append(to_seconds(0, h, m))
    
    # Filter valid candidates
    valid_candidates = [c for c in candidates if c <= video_duration]
    
    if valid_candidates:
        # Return the largest valid timestamp
        best = max(valid_candidates)
    else:
        # Clamp to video duration
        best = video_duration
    
    return to_hms(best)

# Test the function
if __name__ == "__main__":
    # Test cases
    from icecream import ic
    ic(timestamp_to_seconds("01:10:08.45"))
    ic(fix_timestamp("01:10:00", 300))   # 5 min video, should become 00:01:10
    ic(fix_timestamp("10:30", 300))      # Could be 10:30 or 00:10:30
    ic(fix_timestamp("02:30:00", 7200))  # 2 hour video, original should be fine
    ic(fix_timestamp("01:10:00", 300))   # 5 min video, should become 00:01:10
    ic(fix_timestamp("10:30", 300))      # Could be 10:30 or 00:10:30
    ic(fix_timestamp("02:00:00", 7200))  # 2 hour video, original should be fine