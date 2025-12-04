from __future__ import annotations

import base64
import copy
import hashlib
import logging
import math
import os
import pickle
import sys
import time
import warnings
import re
from functools import lru_cache
from io import BytesIO
from typing import Optional, Tuple, List, Dict, Any, Union

import requests
import numpy as np
import torch
from packaging import version
from PIL import ImageFile, ImageOps, Image


logger = logging.getLogger(__name__)

MAX_FRAMES: int = 128
FRAME_SAMPLE_MODE: str = "uniform_last_frame"
MAX_VIDEO_FPS: float = 8.0
SAMPLING_FPS: float = 2.0
CANDIDATE_SAMPLING_FPS: Tuple[float] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
MAX_FPS: Union[float, None, Tuple[Optional[float]]] = 2

# Video caching configuration
MOLMO2_VIDEO_CACHE_DIR = os.environ.get("MOLMO2_VIDEO_CACHE_DIR", "data/video_cache_molmo2")
os.makedirs(MOLMO2_VIDEO_CACHE_DIR, exist_ok=True)


def fetch_single_image(image: str | Image.Image) -> Image.Image:
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        # fix memory leak issue while using BytesIO
        with requests.get(image, stream=True) as response:
            response.raise_for_status()
            with BytesIO(response.content) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            # fix memory leak issue while using BytesIO
            with BytesIO(data) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
    else:
        image_obj = Image.open(image)
    
    if image_obj is None:
        raise ValueError(
            f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}"
        )
    
    with warnings.catch_warnings(record=True) as w:
        image_obj = image_obj.convert("RGB")
    try:
        image_obj = ImageOps.exif_transpose(image_obj)
    except Exception as e:
        pass

    return image_obj


def fetch_image(ele: Dict[str, Any]) -> Image.Image | List[Image.Image]:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    if isinstance(image, (list, tuple)):
        return [fetch_single_image(img) for img in image]
    else:
        return fetch_single_image(image)


def is_decord_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("decord") is not None


def is_torchcodec_available() -> bool:
    """Check if torchcodec is available and properly installed."""
    try:
        import importlib.util
        if importlib.util.find_spec("torchcodec") is None:
            return False
        from torchcodec.decoders import VideoDecoder
        return True
    except (ImportError, AttributeError, Exception):
        return False


@lru_cache(maxsize=1)
def get_default_video_reader_backend() -> str:
    if is_torchcodec_available():
        video_reader_backend = "torchcodec"
    elif is_decord_available():
        video_reader_backend = "decord"
    else:
        video_reader_backend = "av"
    print(f"molmo-utils using {video_reader_backend} by default to read video.", file=sys.stderr)
    return video_reader_backend


def get_candidate_sampling_fps(
    video_fps: float,
    sampling_fps: float,
    max_fps: float = MAX_VIDEO_FPS,
) -> List[float]:
    """
    Return the subset of `video_fps` factors that remain multiples of `sampling_fps`.

    Examples:
        >>> get_candidate_sampling_fps(video_fps=6, sampling_fps=2)
        [2, 6]
        >>> get_candidate_sampling_fps(video_fps=5, sampling_fps=1)
        [1, 5]
        >>> get_candidate_sampling_fps(video_fps=2, sampling_fps=2)
        [2]
        >>> get_candidate_sampling_fps(video_fps=5, sampling_fps=2)
        Traceback (most recent call last):
            ...
        ValueError: sampling_fps=2 must divide video_fps=5 to produce consistent frame steps.
    """
    video_fps = int(video_fps)
    sampling_fps = int(sampling_fps)
    max_fps = int(max_fps)

    if sampling_fps is None:
        raise ValueError("sampling_fps must be provided")
    if video_fps <= 0 or sampling_fps <= 0:
        raise ValueError(f"video_fps and sampling_fps must be positive (got {video_fps}, {sampling_fps})")
    if video_fps % sampling_fps != 0:
        raise ValueError(f"sampling_fps={sampling_fps} must divide video_fps={video_fps}.")

    candidates = []
    for candidate in range(sampling_fps, video_fps + 1, sampling_fps):
        if candidate > max_fps:
            break
        if video_fps % candidate == 0:
            candidates.append(float(candidate))
    return candidates


def get_sampling_fps(
    video_fps: float,
    max_frames: int,
    total_frames: int,
    frame_sample_mode: str,
    candidate_sampling_fps: Tuple[float],
) -> float:
    """
    Get the sampling fps that best spans the video and has the most frames sampled
    """
    num_frames_sampled = 0
    selected_sampling_fps = None
    for sampling_fps in candidate_sampling_fps:
        step_size = max(int(video_fps / sampling_fps), 1)
        num_frames_sampled_at_fps = int(total_frames / step_size)
        if num_frames_sampled == 0:
            if "uniform" in frame_sample_mode:
                if num_frames_sampled_at_fps > max_frames:
                    break
            selected_sampling_fps = sampling_fps
            num_frames_sampled = num_frames_sampled_at_fps

        else:
            # the candidate sampling fps increases so frame count can't decrease
            assert num_frames_sampled <= num_frames_sampled_at_fps
            if num_frames_sampled_at_fps > max_frames:
                # choose the sampling fps that spans the video
                continue

            elif num_frames_sampled_at_fps > num_frames_sampled:
                # both are less than max_frames, choose the one with higher density of frames sampled
                selected_sampling_fps = sampling_fps
                num_frames_sampled = num_frames_sampled_at_fps
    return selected_sampling_fps


def get_frame_times_and_chosen_fps(selected_sampling_fps, total_frames, max_frames, video_fps):
    if selected_sampling_fps is None:
        frame_indices = np.linspace(0, total_frames, max_frames, endpoint=False, dtype=int)
    else:
        step_size = max(int(video_fps / selected_sampling_fps), 1)
        frame_indices = np.arange(0, total_frames, step_size)
    if len(frame_indices) > max_frames:
        frame_indices = frame_indices[:max_frames]
    return selected_sampling_fps, frame_indices


def sample_times(
    duration: float,
    max_frames: int,
    frame_sample_mode: str,
    candidate_sampling_fps: Tuple[float],
    max_fps: Union[float, None, Tuple[Optional[float]]] = None,
    is_training: bool = False,
) -> Tuple[float, np.ndarray, Optional[str]]:

    if frame_sample_mode == "uniform":
        if max_fps:
            raise NotImplementedError("Max FPS with uniform")
        times = np.linspace(0, duration, num=max_frames, endpoint=False, dtype=np.float64)
        return None, times, None
    if frame_sample_mode in ["uniform_last_frame", "uniform_last_frame_sample_fps"]:
        if frame_sample_mode == "uniform_last_frame_sample_fps":
            start, end = max_fps
            if is_training:
                if np.random.random() < 0.1:
                    max_fps = start
                else:
                    max_fps = np.random.uniform(start, end)
            else:
                max_fps = start
        elif isinstance(max_fps, (tuple, list)):
            if is_training and len(max_fps) > 1:
                max_fps = max_fps[np.random.randint(len(max_fps))]
            else:
                max_fps = max_fps[0]
        
        if max_fps is not None:
            max_duration = (max_frames-1) / max_fps  # -1 to include the last frame
            if max_duration < duration:
                times = np.linspace(0, duration, num=max_frames, endpoint=True, dtype=np.float64)
            else:
                times = np.arange(0.0, stop=duration, step=1/max_fps)
                times = np.concatenate([times, [duration]], axis=0)
                assert len(times) <= max_frames
        else:
            times = np.linspace(0, duration, num=max_frames, endpoint=True, dtype=np.float64)
        return None, times, None
    elif frame_sample_mode == "fps":
        # Try larger and larger FPSs until we hit one that can't span the video
        sampling_fps = candidate_sampling_fps[0]
        for candidate_fps in candidate_sampling_fps[1:]:
            if max_frames/candidate_fps < duration:
                break
            sampling_fps = candidate_fps
        times = np.arange(0, max_frames) / sampling_fps
        times = times[times < duration]
        return sampling_fps, times, None
    else:
        raise NotImplementedError(frame_sample_mode)


def sample_frames(
    video_fps: float,
    total_frames: int,
    max_frames: int,
    frame_sample_mode: str,
    candidate_sampling_fps: Tuple[float],
    min_fps: Optional[float] = None,
    is_training: bool = False,
) -> Tuple[float, np.ndarray, Optional[str]]:
    assert total_frames > 0
    rng = np.random
    if frame_sample_mode == "uniform":
        times = np.linspace(0, total_frames, num=min(max_frames, total_frames), endpoint=False, dtype=np.int32)
        return None, times, None
    elif (
        frame_sample_mode.startswith("uniform_last_frame_min_") or
        frame_sample_mode.startswith("uniform_last_frame_max_fps_set_") or
        (frame_sample_mode == "uniform_last_frame" and min_fps is not None)
    ):
        if frame_sample_mode == "uniform_last_frame":
            pass
        elif frame_sample_mode.startswith("uniform_last_frame_max_fps_set_"):
            options = frame_sample_mode.split("uniform_last_frame_max_fps_set_")[1].split("-")
            options = [float(x) for x in options]
            if not is_training:
                min_fps = options[0]  # 0th option is eval default
            else:
                min_fps = np.random.choice(options)
        # These other cases are for backwards-compatibility
        elif frame_sample_mode.startswith("uniform_last_frame_min_2-4"):
            if not is_training:
                min_fps = 2
            else:
                min_fps = 2 if (np.random.random() > 0.5) else 4
        elif frame_sample_mode.startswith("uniform_last_frame_min_2-6"):
            if not is_training:
                min_fps = 2
            else:
                r = np.random.random()
                if r < 0.333:
                    min_fps = 2
                elif r < 0.666:
                    min_fps = 4
                else:
                    min_fps = 6
        else:
            min_fps = float(re.fullmatch("uniform_last_frame_min_([0-9\.]+)fps", frame_sample_mode).group(1))
        
        duration = total_frames / video_fps
        if total_frames <= 2:
            return None, np.arange(total_frames, dtype=np.int64), None
        if duration > (max_frames/min_fps - 1):  # -1 for first and last frame
            # uniform fallback
            times = np.linspace(0, total_frames-1, num=min(max_frames, total_frames), endpoint=True, dtype=np.int32)
            return None, times, None
        else:
            float_indices = np.arange(0.0, stop=total_frames-1, step=float(video_fps/min_fps))
            if np.round(float_indices[-1]) != total_frames-1:
                float_indices = np.concatenate([float_indices, [total_frames-1]], axis=0)
            indices = np.round(float_indices)
            assert indices[-1] < total_frames
            assert len(float_indices) <= max_frames
            return min_fps, indices.astype(np.int32), None
    elif frame_sample_mode == "uniform_last_frame":
        times = np.linspace(0, total_frames-1, num=min(max_frames, total_frames), endpoint=True, dtype=np.int32)
        return None, times, None
    elif frame_sample_mode == "uniform_randomized":
        aug = None
        if total_frames <= max_frames:
            indices = np.arange(0, total_frames, dtype=np.int32)
            step = 1
        else:
            step = total_frames // max_frames
            indices = np.arange(0, max_frames, dtype=np.int32) * step
            remainder = (total_frames - step * (max_frames - 1))
            if rng.random() > 0.3 and is_training:
                offset = rng.randint(-(step//2), remainder//2)
                indices[1:] += offset
                assert indices[1] > indices[0] and indices[-1] < total_frames
                aug = "RE"
            return None, indices, aug
    elif frame_sample_mode in ["fps", "fps_uniform"]:
        selected_sampling_fps = get_sampling_fps(
            video_fps, max_frames, total_frames, frame_sample_mode, candidate_sampling_fps
        )
        sampling_fps, frame_indices = get_frame_times_and_chosen_fps(
            selected_sampling_fps, total_frames, max_frames, video_fps
        )
        return selected_sampling_fps, frame_indices, None
    else:
        raise NotImplementedError(frame_sample_mode)


def _validate_clip(clip: Tuple[float, float], duration: float):
    if clip[0] >= clip[1]:
        raise ValueError(f"Clip {clip} has start>=end")
    if clip[0] >= duration:
        raise ValueError(f"Invalid clip, start={clip[0]} but video duration={duration}")


def load_video_decord(ele: Dict[str, Any]) -> Dict[str, Any]:
    """load video using decord.VideoReader

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - clip (tuple): the start and end time of clip.
            - max_frames (int): the max number of frames.
            - frame_sample_mode (str): the mode of frame sampling.
            - sampling_fps (Optional[float]): Rate to sample points at, in frames per second.
                Used for `frame_sample_mode` == "fps"
            - candidate_sampling_fps (tuple): the candidate sampling fps.
            - min_fps (Optional[float]): the minimum fps to sample. Default to None.
            - max_fps (Optional[float]): the maximum fps to sample. Default to 2.
            - time_sampling (bool): whether to use time sampling. Default to False.
            - is_training (bool): whether to use training mode.
    Returns:
        a dict contains the sampled frames and metadata.
            - frames (np.ndarray): the sampled frames as RGB numpy array.
            - timestamps (np.ndarray): the timestamps of the sampled frames.
            - target_fps (Optional[float]): the target fps of the sampled frames, if there was one
            - sampling_augmentation (Optional[str]): the augmentation used.
    """
    import decord
    video_path = ele["video"]
    clip = ele.get("clip", None)
    max_frames = ele.get("max_frames", MAX_FRAMES)
    frame_sample_mode = ele.get("frame_sample_mode", FRAME_SAMPLE_MODE)
    min_fps = ele.get("min_fps", None)
    max_fps = ele.get("max_fps", MAX_FPS)
    time_sampling = ele.get("time_sampling", False)
    is_training = ele.get("is_training", False)
    vr = decord.VideoReader(video_path, num_threads=1, ctx=decord.cpu(0))
    video_fps = vr.get_avg_fps()
    if frame_sample_mode == "fps":
        sampling_fps = ele.get("sampling_fps", SAMPLING_FPS)
        candidate_sampling_fps = get_candidate_sampling_fps(video_fps, sampling_fps)
    else:
        candidate_sampling_fps = ele.get("candidate_sampling_fps", CANDIDATE_SAMPLING_FPS)
    if time_sampling:
        time_stamps = vr.get_frame_timestamp(list(range(len(vr))))
        duration = time_stamps[-1][1] - time_stamps[0][0]
        if clip:
            _validate_clip(clip, duration)
            clip_duration = min(clip[1], duration) - clip[0]
            target_fps, target_timestamps, aug = sample_times(
                clip_duration,
                max_frames,
                frame_sample_mode,
                candidate_sampling_fps,
                max_fps,
                is_training,
            )
        else:
            target_fps, target_timestamps, aug = sample_times(
                duration,
                max_frames,
                frame_sample_mode,
                candidate_sampling_fps,
                max_fps,
                is_training,
            )
        target_timestamps = np.array(target_timestamps) + time_stamps[0, 0]
        ix = np.searchsorted(time_stamps[:, 1], target_timestamps)
        # In case of minor floating point errors
        ix = np.minimum(ix, len(time_stamps) - 1)
        frames = vr.get_batch(ix).asnumpy()
        return dict(
            frames=frames,
            timestamps=np.array(target_timestamps),
            target_fps=-1,
            sampling_augmentation="" if aug is None else aug,
        )
    else:
        if clip:
            _validate_clip(clip, len(vr)/video_fps)
            start_frame = math.floor(clip[0] * video_fps)
            end_frame = min(math.ceil(clip[1] * video_fps), len(vr))
            sampling_fps, frame_indices, aug = sample_frames(
                video_fps,
                end_frame-start_frame,
                max_frames,
                frame_sample_mode,
                candidate_sampling_fps,
                min_fps,
                is_training,
            )
        else:
            sampling_fps, frame_indices, aug = sample_frames(
                video_fps,
                len(vr),
                max_frames,
                frame_sample_mode,
                candidate_sampling_fps,
                min_fps,
                is_training,
            )
            start_frame = 0
        frames = vr.get_batch(frame_indices + start_frame).asnumpy()
        return dict(
            frames=frames,
            timestamps=np.array(frame_indices)/video_fps,
            target_fps=-1 if sampling_fps is None else sampling_fps,
            sampling_augmentation="" if aug is None else aug,
        )


def load_video_torchcodec(ele: Dict[str, Any]) -> Dict[str, Any]:
    """load video using torchcodec.VideoDecoder

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - clip (tuple): the start and end time of clip.
            - max_frames (int): the max number of frames.
            - frame_sample_mode (str): the mode of frame sampling.
            - sampling_fps (Optional[float]): Rate to sample points at, in frames per second.
                Used for `frame_sample_mode` == "fps".
            - candidate_sampling_fps (tuple): the candidate sampling fps.
            - min_fps (Optional[float]): the minimum fps to sample. Default to None.
            - max_fps (Optional[float]): the maximum fps to sample. Default to 2.
            - time_sampling (bool): whether to use time sampling. Default to True.
            - is_training (bool): whether to use training mode.
    Returns:
        a dict contains the sampled frames and metadata.
            - frames (np.ndarray): the sampled frames as RGB numpy array.
            - timestamps (np.ndarray): the timestamps of the sampled frames.
            - target_fps (Optional[float]): the target fps of the sampled frames, if there was one
            - sampling_augmentation (Optional[str]): the augmentation used.
    """
    import torchcodec
    video_path = ele["video"]
    clip = ele.get("clip", None)
    max_frames = ele.get("max_frames", MAX_FRAMES)
    frame_sample_mode = ele.get("frame_sample_mode", FRAME_SAMPLE_MODE)
    min_fps = ele.get("min_fps", None)
    max_fps = ele.get("max_fps", MAX_FPS)
    time_sampling = ele.get("time_sampling", True)
    is_training = ele.get("is_training", False)
    decoder = torchcodec.decoders.VideoDecoder(video_path, num_ffmpeg_threads=1, device="cpu")
    video_fps = decoder.metadata.average_fps
    total_frames = decoder.metadata.num_frames

    if frame_sample_mode == "fps":
        sampling_fps = ele.get("sampling_fps", SAMPLING_FPS)
        candidate_sampling_fps = get_candidate_sampling_fps(video_fps, sampling_fps)
    else:
        candidate_sampling_fps = ele.get("candidate_sampling_fps", CANDIDATE_SAMPLING_FPS)

    if time_sampling:
        # If the first frame starts at > 0, we effectively clip the video starting at that time
        # since (most) video players would also skip to that time
        time_offset = decoder.metadata.begin_stream_seconds_from_content
        # Note this duration does assume we started playing at `time_offset`
        duration = decoder.metadata.duration_seconds

        if clip:
            _validate_clip(clip, duration)
            clip_duration = min(clip[1], duration) - clip[0]
            target_fps, target_timestamps, aug = sample_times(
                clip_duration,
                max_frames,
                frame_sample_mode,
                candidate_sampling_fps,
                max_fps,
                is_training,
            )
            time_offset += clip[0]
        else:
            target_fps, target_timestamps, aug = sample_times(
                duration,
                max_frames,
                frame_sample_mode,
                candidate_sampling_fps,
                max_fps,
                is_training,
            )

        # Floating point/rounding issues might cause `target_timestamps` to be very slightly
        # out-of-bounds, to handle this we sanity check then clip them
        assert all(x >= 0 for x in target_timestamps)
        assert all(x < duration+1e-6 for x in target_timestamps)
        # 1e-6 padding since torchcodec can throw out-of-bounds errors even if you ask for the
        # exact boundary value, we should still get the first/last frame anyway
        max_timestamp = decoder.metadata.end_stream_seconds_from_content - 1e-6
        min_timestamp = decoder.metadata.begin_stream_seconds_from_content + 1e-6
        # Note we avoid using numpy ops here to reduce floating precision issues
        timestamps = [x + time_offset for x in target_timestamps]
        timestamps = [max(min_timestamp, min(max_timestamp, x)) for x in timestamps]
        frames = decoder.get_frames_played_at(timestamps)
        target_timestamps = np.array(target_timestamps)
    else:
        if clip is not None:
            duration = total_frames / video_fps
            _validate_clip(clip, duration)
            start_index = math.floor(clip[0] * video_fps)
            end_index = min(math.ceil(clip[1] * video_fps), total_frames)
            target_fps, indices, aug = sample_frames(
                video_fps,
                end_index-start_index,
                max_frames,
                frame_sample_mode,
                candidate_sampling_fps,
                min_fps,
                is_training,
            )
        else:
            start_index = 0
            target_fps, indices, aug = sample_frames(
                video_fps,
                total_frames,
                max_frames,
                frame_sample_mode,
                candidate_sampling_fps,
                min_fps,
                is_training,
            )
        frames = decoder.get_frames_at(indices=indices+start_index)
        target_timestamps = np.array(indices) / video_fps
    
    return dict(
        frames=frames.data.numpy().transpose(0, 2, 3, 1),  # Convert to THWC format
        timestamps=target_timestamps,
        target_fps=-1 if target_fps is None else target_fps,
        sampling_augmentation="" if aug is None else aug,
    )


def load_video_av_noseek(ele: Dict[str, Any]) -> Dict[str, Any]:
    """Load a video frames by decoding all frames with pyav

    More robust than `load_video_decord` but can be much slower for long videos
    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - clip (tuple): the start and end time of clip.
            - max_frames (int): the max number of frames.
            - frame_sample_mode (str): the mode of frame sampling.
            - sampling_fps (Optional[float]): Rate to sample points at, in frames per second.
                Used for `frame_sample_mode` == "fps"
            - candidate_sampling_fps (tuple): the candidate sampling fps.
            - min_fps (Optional[float]): the minimum fps to sample. Default to None.
            - is_training (bool): whether to use training mode.
    Returns:
        a dict contains the sampled frames and metadata.
            - frames (np.ndarray): the sampled frames as RGB numpy array.
            - timestamps (np.ndarray): the timestamps of the sampled frames.
            - target_fps (Optional[float]): the target fps of the sampled frames, if there was one
            - sampling_augmentation (Optional[str]): the augmentation used.
    """
    import av
    video_path = ele["video"]
    if "http://" in video_path or "https://" in video_path:
        raise ValueError("av does not support http/https video path")
    if "file://" in video_path:
        video_path = video_path[7:]
    clip = ele.get("clip", None)
    max_frames = ele.get("max_frames", MAX_FRAMES)
    frame_sample_mode = ele.get("frame_sample_mode", FRAME_SAMPLE_MODE)
    candidate_sampling_fps = ele.get("candidate_sampling_fps", CANDIDATE_SAMPLING_FPS)
    min_fps = ele.get("min_fps", None)
    time_sampling = ele.get("time_sampling", False)
    is_training = ele.get("is_training", False)
    if time_sampling:
        raise NotImplementedError("time_sampling is not supported for av")

    # Behaves the same as the old version using `imageio.v3` but avoid extra the dependency
    with av.open(video_path) as container:
        video_stream = container.streams.video[0]
        fps = video_stream.guessed_rate
        if frame_sample_mode == "fps":
            sampling_fps = ele.get("sampling_fps", SAMPLING_FPS)
            candidate_sampling_fps = get_candidate_sampling_fps(fps, sampling_fps)
        it = container.decode(video=0)
        frames = list(it)
        if clip is not None:
            duration = len(frames) / fps
            _validate_clip(clip, duration)
            start_index = math.floor(clip[0] * fps)
            end_index = min(math.ceil(clip[1] * fps), len(frames))
            target_fps, indices, aug = sample_frames(
                fps,
                end_index-start_index,
                max_frames,
                frame_sample_mode,
                candidate_sampling_fps,
                min_fps,
                is_training,
            )
        else:
            start_index = 0
            target_fps, indices, aug = sample_frames(
                fps,
                len(frames),
                max_frames,
                frame_sample_mode,
                candidate_sampling_fps,
                min_fps,
                is_training,
            )
        frames = [frames[i+start_index].to_ndarray(format="rgb24", channel_last=True) for i in indices]
        return dict(
            frames=np.stack(frames, axis=0),
            timestamps=np.array(indices)/float(fps),
            target_fps=-1 if target_fps is None else target_fps,
            sampling_augmentation="" if aug is None else aug,
        )


VIDEO_READER_BACKENDS = {
    "decord": load_video_decord,
    "torchcodec": load_video_torchcodec,
    "av": load_video_av_noseek,
}


def _get_cache_key(video_path: str, ele: Dict[str, Any]) -> str:
    """Generate a unique cache key based on video path and processing parameters."""
    # Normalize video path for cache key
    normalized_path = video_path
    if video_path.startswith("file://"):
        normalized_path = video_path[7:]
    
    # Generate video basename
    if video_path.startswith("http://") or video_path.startswith("https://"):
        # For URLs, use the full URL (hashed) to avoid collisions
        url_hash = hashlib.md5(video_path.encode()).hexdigest()[:16]
        video_basename = f"url_{url_hash}"
    else:
        video_basename = os.path.basename(normalized_path).split(".")[0]
        if not video_basename:  # Fallback if basename extraction fails
            video_basename = hashlib.md5(normalized_path.encode()).hexdigest()[:16]
    
    # Include relevant processing parameters in the key
    params = []
    for key in ["clip", "max_frames", "frame_sample_mode", "sampling_fps", 
                "candidate_sampling_fps", "min_fps", "max_fps", "time_sampling", 
                "is_training", "backend"]:
        if key in ele:
            value = ele[key]
            # Convert tuples/lists to strings for hashing
            if isinstance(value, (tuple, list)):
                value = "_".join(str(v) for v in value)
            params.append(f"{key}={value}")
    
    if params:
        param_str = "_".join(params)
        # Hash long parameter strings to keep filename manageable
        if len(param_str) > 50:
            param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
            return f"{video_basename}_{param_hash}"
        return f"{video_basename}_{param_str}"
    
    return video_basename


def _validate_cached_data(cached_data: Dict[str, Any]) -> bool:
    """
    Validate that cached data contains all required keys and valid data.
    
    Args:
        cached_data: Dictionary containing cached video data
        
    Returns:
        bool: True if cache is valid, False otherwise
    """
    if not isinstance(cached_data, dict):
        logger.warning("Cached data is not a dictionary")
        return False
    
    # Check for required keys
    required_keys = ["frames", "timestamps", "target_fps", "sampling_augmentation"]
    for key in required_keys:
        if key not in cached_data:
            logger.warning(f"Cached data missing key: {key}")
            return False
    
    # Validate frames
    frames = cached_data.get("frames")
    if not isinstance(frames, np.ndarray):
        logger.warning("Cached frames is not a numpy array")
        return False
    
    if frames.ndim != 4:  # Should be (T, H, W, C)
        logger.warning(f"Cached frames has invalid dimensions: {frames.shape}")
        return False
    
    # Validate timestamps
    timestamps = cached_data.get("timestamps")
    if not isinstance(timestamps, np.ndarray):
        logger.warning("Cached timestamps is not a numpy array")
        return False
    
    if len(timestamps) != len(frames):
        logger.warning(f"Cached timestamps length ({len(timestamps)}) doesn't match frames length ({len(frames)})")
        return False
    
    # Validate target_fps
    target_fps = cached_data.get("target_fps")
    if not isinstance(target_fps, (int, float)) and target_fps != -1:
        logger.warning(f"Cached target_fps is invalid: {target_fps}")
        return False
    
    return True


def _load_cached_video(cache_filepath: str) -> Optional[Dict[str, Any]]:
    """Load cached video data from disk with validation."""
    try:
        with open(cache_filepath, "rb") as f:
            data = pickle.load(f)
        
        # Validate the cached data
        if not _validate_cached_data(data):
            logger.warning(f"Cached data validation failed for {cache_filepath}, will re-process")
            # Delete corrupt cache file
            try:
                os.remove(cache_filepath)
                logger.info(f"Removed corrupt cache file: {cache_filepath}")
            except Exception as e:
                logger.warning(f"Failed to remove corrupt cache file: {e}")
            return None
        
        logger.info(f"Video loaded from cache: {cache_filepath}")
        return data
    except (pickle.UnpicklingError, EOFError, AttributeError) as e:
        logger.warning(f"Failed to unpickle cache file {cache_filepath}: {e}, will re-process")
        # Delete corrupt cache file
        try:
            os.remove(cache_filepath)
            logger.info(f"Removed corrupt cache file: {cache_filepath}")
        except Exception as e:
            logger.warning(f"Failed to remove corrupt cache file: {e}")
        return None
    except Exception as e:
        logger.debug(f"Failed to load video from cache {cache_filepath}: {e}")
        return None


def _save_cached_video(cache_filepath: str, video_data: Dict[str, Any]):
    """Save processed video data to cache."""
    try:
        # Create temporary file first, then rename (atomic operation on most systems)
        temp_filepath = cache_filepath + ".tmp"
        with open(temp_filepath, "wb") as f:
            pickle.dump(video_data, f)
        
        # Rename temp file to actual cache file
        os.replace(temp_filepath, cache_filepath)
        logger.info(f"Video saved to cache: {cache_filepath}")
    except Exception as e:
        logger.error(f"Failed to save video to cache: {e}")
        # Clean up temp file if it exists
        temp_filepath = cache_filepath + ".tmp"
        if os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
            except Exception:
                pass


def fetch_video(ele: Dict[str, Any], use_cache: bool = True) -> Dict[str, Any]:
    # Handle string video path (file/url)
    if isinstance(ele["video"], str):
        video_input = None
        
        # Check cache first if enabled
        if use_cache:
            cache_key = _get_cache_key(ele["video"], ele)
            cache_filepath = os.path.join(MOLMO2_VIDEO_CACHE_DIR, f"{cache_key}.pkl")
            
            if os.path.exists(cache_filepath):
                cached_data = _load_cached_video(cache_filepath)
                if cached_data is not None:
                    video_input = cached_data
        
        # Process video if not loaded from cache
        if video_input is None:
            video_reader_backend = ele.get("backend", get_default_video_reader_backend())
            if video_reader_backend == "torchcodec":
                video_input = load_video_torchcodec(ele)
            else:
                try:
                    video_input = VIDEO_READER_BACKENDS[video_reader_backend](ele)
                except Exception as e:
                    video_input = VIDEO_READER_BACKENDS["av"](ele)
            
            # Save to cache if enabled
            if use_cache:
                cache_key = _get_cache_key(ele["video"], ele)
                cache_filepath = os.path.join(MOLMO2_VIDEO_CACHE_DIR, f"{cache_key}.pkl")
                _save_cached_video(cache_filepath, video_input)
    else:
        # Handle list/tuple of images (no caching for this case)
        assert isinstance(ele["video"], (list, tuple))
        frames = [fetch_single_image(img) for img in ele["video"]]
        timestamps = ele.get("timestamps", None)
        if timestamps is None:
            raise ValueError("timestamps is required when video is a list of images")
        video_input = dict(
            frames=np.stack(frames, axis=0),
            timestamps=timestamps,
            target_fps=-1,
            sampling_augmentation="",
        )
    return video_input


def extract_vision_info(
    conversations: List[Dict] | List[List[Dict]]
):
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], (list, tuple)):
                for ele in message["content"]:
                    if (
                        ("image" in ele and ele.get("type", "") in ("image", "image_url"))
                        or ("video" in ele and ele.get("type", "") in ("video",))
                    ):
                        vision_infos.append(ele)
    return vision_infos


def process_vision_info(
    conversations: List[Dict] | List[List[Dict]],
    use_cache: bool = False,
):
    vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info))
        elif "video" in vision_info:
            video_inputs.append(fetch_video(vision_info, use_cache=use_cache))
        else:
            raise ValueError("image, image_url or video should be in content")
    
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    return image_inputs, video_inputs