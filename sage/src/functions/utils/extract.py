import os
import shutil
import cv2
import subprocess
from tqdm import tqdm
from typing import List, Dict
from sage.src.functions.utils.temporal import seconds_to_timestamp, get_video_duration, fix_timestamp, timestamp_to_seconds


def _setup_output_directory(
    video_path: str, extract_type: str, timestamp_start: float, timestamp_end: float
) -> tuple[str, str]:
    """Setup output directory for extraction."""
    video_basename = video_path.split("/")[-1].split(".")[0]
    video_dir = os.path.dirname(video_path)
    output_dir = f"{video_dir}/{video_basename}_{extract_type}_{timestamp_start}_{timestamp_end}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir, video_basename


def _get_frame_indices(start_frame: int, end_frame: int, fps: float, max_frames: int = 10) -> List[int]:
    """Get frame indices to extract, limiting to max_frames if needed."""
    frame_indices = [frame_num for frame_num in range(start_frame, end_frame + 1, int(fps))]
    if len(frame_indices) > max_frames:
        new_frame_indices = []
        for i in range(max_frames):
            new_frame_indices.append(frame_indices[i * len(frame_indices) // max_frames])
        frame_indices = new_frame_indices
    return frame_indices


def extract_frames(
    video_path: str,
    timestamp_start: float,
    timestamp_end: float,
    num_frames: int = 10,
) -> List[str]:
    """Efficiently extract frames from video between timestamps."""
    output_dir, video_basename = _setup_output_directory(
        video_path,
        "frames",
        seconds_to_timestamp(timestamp_start, in_mins=True),
        seconds_to_timestamp(timestamp_end, in_mins=True),
    )

    # We need fps to compute start/end frames
    cap_tmp = cv2.VideoCapture(video_path)
    if not cap_tmp.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    fps = cap_tmp.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap_tmp.release()
        raise ValueError(f"Invalid FPS ({fps}) for video: {video_path}")
    start_frame = int(timestamp_start * fps)
    end_frame = int(timestamp_end * fps)
    frame_indices = [
        start_frame + int(i * (end_frame - start_frame) / (num_frames - 1)) for i in range(num_frames)
    ]
    cap_tmp.release()
    
    # Precompute all expected frame paths
    expected_frame_paths = []
    for current_frame in frame_indices:
        timestamp_str = seconds_to_timestamp(current_frame / fps, in_mins=True)
        frame_path = os.path.join(
            output_dir,
            f"{video_basename}_frame_{current_frame}_{timestamp_str}.jpg",
        )
        expected_frame_paths.append(frame_path)

    # If all frames exist, return immediately
    if all(os.path.exists(p) for p in expected_frame_paths):
        return expected_frame_paths

    # Otherwise, proceed to extract missing frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise ValueError(f"Invalid FPS ({fps}) for video: {video_path}")
    start_frame = int(timestamp_start * fps)
    end_frame = int(timestamp_end * fps)
    frame_set = set(frame_indices)
    saved_frame_paths = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if current_frame in frame_set:
            timestamp_str = seconds_to_timestamp(current_frame / fps, in_mins=True)
            frame_path = os.path.join(
                output_dir,
                f"{video_basename}_frame_{current_frame}_{timestamp_str}.jpg",
            )
            if not os.path.exists(frame_path):
                cv2.imwrite(frame_path, frame)
            saved_frame_paths.append(frame_path)
        current_frame += 1
    cap.release()
    return saved_frame_paths

import os
import subprocess

def extract_subclip(video_path: str, timestamp_start: float, timestamp_end: float) -> str:
    """
    Robustly extracts a subclip from a video.
    1. Tries a fast, brittle stream copy.
    2. If it fails or times out, falls back to a slower, more robust re-encode.
    3. Returns the original video path if all attempts fail.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return video_path

    output_dir = os.path.dirname(video_path)
    base = os.path.splitext(os.path.basename(video_path))[0]
    output_filename = f"{base}_subclip_{seconds_to_timestamp(timestamp_start)}_{seconds_to_timestamp(timestamp_end)}.mp4"
    output_path = os.path.join(output_dir, output_filename)

    if os.path.exists(output_path):
        return output_path

    duration = timestamp_end - timestamp_start
    if duration <= 0:
        print("Error: End timestamp must be after start timestamp.")
        return video_path

    # --- Attempt 1: Fast Stream Copy (Prone to hanging on fragile files) ---
    print(f"Attempting fast stream copy for: {video_path}")
    fast_command = [
        "ffmpeg", "-y", "-ss", str(timestamp_start), "-i", video_path,
        "-t", str(duration), "-c", "copy", "-avoid_negative_ts", "make_zero",
        output_path
    ]

    try:
        # Run with a shorter timeout. Stream copy should be fast.
        result = subprocess.run(
            fast_command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, # Discard stderr to avoid hangs
            timeout=60  # 60 seconds should be plenty for a copy
        )
        if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
            # print(f"Successfully created subclip with fast copy: {output_path}")
            return output_path
        print("Fast stream copy failed or created an empty file. Trying fallback.")

    except subprocess.TimeoutExpired:
        print("Fast stream copy timed out. This often indicates a fragile video file or high I/O load.")
        print("Falling back to a full re-encode, which will be slower.")
    except Exception as e:
        print(f"An unexpected error occurred during fast copy: {e}. Trying fallback.")


    # --- Attempt 2: Slow Re-encoding (More robust) ---
    print(f"Attempting fallback re-encoding for: {video_path}")
    slow_command = [
        "ffmpeg", "-y", "-ss", str(timestamp_start), "-i", video_path,
        "-t", str(duration), "-c:v", "libx264", "-preset", "fast",
        "-c:a", "aac", "-loglevel", "error", output_path
    ]

    try:
        # Give the re-encode much more time.
        result = subprocess.run(
            slow_command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=300 # 5 minutes for re-encoding
        )
        if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
            # print(f"Successfully created subclip with re-encoding: {output_path}")
            return output_path
    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"Fallback re-encoding also failed: {e}")

    # --- Final Failure ---
    print(f"All FFmpeg attempts failed for {video_path}. Returning original video path.")
    if os.path.exists(output_path):
        os.remove(output_path) # Clean up failed attempt
    return video_path