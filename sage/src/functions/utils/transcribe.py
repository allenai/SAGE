import os
from moviepy import VideoFileClip
from sage.utils.utils import API_KEYS
from sage.src.functions.utils.temporal import (
    timestamp_to_seconds,
    seconds_to_timestamp,
)

import os
import sys
import contextlib
import requests
import random

TRANSCRIBE_API_URL = os.environ.get("TRANSCRIBE_API_URL", "None")

def get_random_transcribe_url():
    """
    Get a random Transcribe API URL from comma-separated values.
    If only one URL is provided, return it directly.
    """
    if TRANSCRIBE_API_URL == "None" or not TRANSCRIBE_API_URL:
        return TRANSCRIBE_API_URL
    
    urls = [url.strip() for url in TRANSCRIBE_API_URL.split(",") if url.strip()]
    if not urls:
        return TRANSCRIBE_API_URL
    
    return random.choice(urls)

class DevNull:
    """A file-like object that discards all writes."""
    def write(self, s):
        pass
    
    def flush(self):
        pass
    
    def close(self):
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


@contextlib.contextmanager
def suppress_stdout_stderr():
    """Suppress both stdout and stderr."""
    devnull = DevNull()
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = devnull
        sys.stderr = devnull
        yield
    finally:
        # Ensure we restore the original streams even if there's an exception
        try:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        except:
            pass

def convert_video_to_audio_mp3(
    video_file_path,
    output_audio_file_path,
    timestamp_start: str = None,
    timestamp_end: str = None,
):
    with suppress_stdout_stderr():
        video = VideoFileClip(video_file_path)
    if timestamp_start and timestamp_end:
        timestamp_start_seconds = timestamp_to_seconds(timestamp_start)
        timestamp_end_seconds = timestamp_to_seconds(timestamp_end)
        if (
            timestamp_start_seconds < timestamp_end_seconds
            and timestamp_start_seconds >= 0
            and timestamp_end_seconds <= video.duration
        ):
            video = video.subclipped(timestamp_start_seconds, timestamp_end_seconds)

    audio = video.audio
    if audio is not None:
        audio.write_audiofile(output_audio_file_path, logger=None)
        return output_audio_file_path
    else:
        return None


def transcribe_video(
    filename,
    timestamp_start: str = None,
    timestamp_end: str = None,
):

    audio_file_path = filename.replace("mp4", "mp3")
    if timestamp_start is not None and timestamp_end is not None:
        audio_file_path = audio_file_path.replace(".mp3", f"_{timestamp_start}_{timestamp_end}.mp3")
    
    if not os.path.exists(audio_file_path):
        audio_file_path = convert_video_to_audio_mp3(filename, audio_file_path, timestamp_start, timestamp_end)

    if audio_file_path is None:
        return "Hmm, we might have been given a video with the verbal speech removed, so we cannot transcribe it."

    # Calculate offset if timestamp_start is provided
    offset_seconds = 0
    if timestamp_start:
        offset_seconds = timestamp_to_seconds(timestamp_start)
    
     # save transcript as txt file
    transcript_file_path = audio_file_path.replace(".mp3", ".txt")

    # read txt file if exists
    if os.path.exists(transcript_file_path):
        with open(transcript_file_path, "r") as f:
            transcripts = f.read()
        return transcripts

    try:
        payload = {
            "filepath": audio_file_path,
            "timestamp_start": None,  # timestamps handled in audio slice above
            "timestamp_end": None,
        }
        resp = requests.post(f"{get_random_transcribe_url().rstrip('/')}/transcribe", json=payload, timeout=300)
        resp.raise_for_status()
        result = resp.json()
    except Exception as e:
        raise RuntimeError(f"Transcribe API call failed: {e}")

    # Process segments and apply offset
    transcripts = {}
    for idx, segment in enumerate(result["segments"]):
        adjusted_start = float(segment["start"]) + offset_seconds
        adjusted_end = float(segment["end"]) + offset_seconds
        
        transcript_entry = {
            "text": segment["text"],
            "start": seconds_to_timestamp(adjusted_start, in_mins=True),
            "end": seconds_to_timestamp(adjusted_end, in_mins=True),
        }
        transcripts[idx] = transcript_entry
    
    with open(transcript_file_path, "w") as f:
        f.write(str(transcripts))
    
    return transcripts

if TRANSCRIBE_API_URL != "None":
    from icecream import ic
    ic(transcribe_video("sage/serve/examples/H9Z1xVIhT_I.mp4"))