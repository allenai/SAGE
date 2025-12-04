import base64
import copy
import logging
import math
import os
import pickle
import sys
import time
import warnings
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Optional, Union, Tuple, List, Any, Dict
from concurrent.futures import ThreadPoolExecutor

import requests
import torch
import torchvision
from packaging import version
from PIL import Image
import numpy as np
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode


MAX_RATIO = 200
SPATIAL_MERGE_SIZE = 2
IMAGE_MIN_TOKEN_NUM = 4
IMAGE_MAX_TOKEN_NUM = 16384
VIDEO_MIN_TOKEN_NUM = int(os.environ.get("MIN_TOKENS_PER_FRAME", 1))
VIDEO_MAX_TOKEN_NUM = int(os.environ.get("MAX_TOKENS_PER_FRAME", 48))

FPS = float(os.environ.get("FPS", 1.0))
FRAME_FACTOR = int(os.environ.get("FRAME_FACTOR", 1))
FPS_MIN_FRAMES = int(os.environ.get("MIN_FRAMES", 1))
FPS_MAX_FRAMES = int(os.environ.get("MAX_FRAMES", 128))
MAX_NUM_WORKERS_FETCH_VIDEO = 8


MODEL_SEQ_LEN = int(float(os.environ.get('MODEL_SEQ_LEN', 128000)))
logger = logging.getLogger(__name__)

# Video caching configuration
QWEN_VL_VIDEO_CACHE_DIR = os.environ.get("QWEN_VL_VIDEO_CACHE_DIR", "data/video_cache_qwen3_vl_fr128")
os.makedirs(QWEN_VL_VIDEO_CACHE_DIR, exist_ok=True)


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(height: int, width: int, factor: int, min_pixels: Optional[int] = None, max_pixels: Optional[int] = None) -> Tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    max_pixels = max_pixels if max_pixels is not None else (IMAGE_MAX_TOKEN_NUM * factor ** 2)
    min_pixels = min_pixels if min_pixels is not None else (IMAGE_MIN_TOKEN_NUM * factor ** 2)
    assert max_pixels >= min_pixels, "The max_pixels of image must be greater than or equal to min_pixels, got {max_pixels} < {min_pixels}"
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def to_rgb(pil_image: Image.Image) -> Image.Image:
      if pil_image.mode == 'RGBA':
          white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
          white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
          return white_background
      else:
          return pil_image.convert("RGB")


def fetch_image(ele: Dict[str, Union[str, Image.Image]], image_patch_size: int = 14) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]

    image_obj = None
    patch_factor = int(image_patch_size * SPATIAL_MERGE_SIZE)
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
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
            with BytesIO(data) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = to_rgb(image_obj)

    ## resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=patch_factor,
        )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels", IMAGE_MIN_TOKEN_NUM * patch_factor ** 2)
        max_pixels = ele.get("max_pixels", IMAGE_MAX_TOKEN_NUM * patch_factor ** 2)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=patch_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height))
    return image


def smart_nframes(
    ele: Dict[str, Any],
    total_frames: int,
    video_fps: Union[int, float],
) -> int:
    """calculate the number of frames for video used for model inputs.

    Args:
        ele (dict): a dict contains the configuration of video.
            support either `fps` or `nframes`:
                - nframes: the number of frames to extract for model inputs.
                - fps: the fps to extract frames for model inputs.
                    - min_frames: the minimum number of frames of the video, only used when fps is provided.
                    - max_frames: the maximum number of frames of the video, only used when fps is provided.
        total_frames (int): the original total number of frames of the video.
        video_fps (int | float): the original fps of the video.

    Raises:
        ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

    Returns:
        int: the number of frames for video used for model inputs.
    """
    assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        fps = ele.get("fps", FPS)
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR)
        nframes = total_frames / video_fps * fps
        if nframes > total_frames:
            logger.warning(f"smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]")
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = floor_by_factor(nframes, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
    return nframes


def _read_video_torchvision(
    ele: Dict[str, Any],
) -> Tuple[torch.Tensor, float]:
    """read video using torchvision.io.read_video

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    video_path = ele["video"]
    if version.parse(torchvision.__version__) < version.parse("0.19.0"):
        if "http://" in video_path or "https://" in video_path:
            warnings.warn("torchvision < 0.19.0 does not support http/https video path, please upgrade to 0.19.0.")
        if "file://" in video_path:
            video_path = video_path[7:]
    st = time.time()
    video, audio, info = io.read_video(
        video_path,
        start_pts=ele.get("video_start", 0.0),
        end_pts=ele.get("video_end", None),
        pts_unit="sec",
        output_format="TCHW",
    )
    total_frames, video_fps = video.size(0), info["video_fps"]
    logger.info(f"torchvision:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long()
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    video = video[idx]

    video_metadata = dict(
        fps=video_fps,
        frames_indices=idx,
        total_num_frames=total_frames,
        video_backend="torchvision",
    )
    return video, video_metadata, sample_fps


def is_decord_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("decord") is not None


def calculate_video_frame_range(
    ele: Dict[str, Any],
    total_frames: int,
    video_fps: float,
) -> Tuple[int, int, int]:
    """
    Calculate the start and end frame indices based on the given time range.

    Args:
        ele (dict): A dictionary containing optional 'video_start' and 'video_end' keys (in seconds).
        total_frames (int): Total number of frames in the video.
        video_fps (float): Frames per second of the video.

    Returns:
        tuple: A tuple containing (start_frame, end_frame, frame_count).

    Raises:
        ValueError: If input parameters are invalid or the time range is inconsistent.
    """
    # Validate essential parameters
    if video_fps <= 0:
        raise ValueError("video_fps must be a positive number")
    if total_frames <= 0:
        raise ValueError("total_frames must be a positive integer")

    # Get start and end time in seconds
    video_start = ele.get("video_start", None)
    video_end = ele.get("video_end", None)
    if video_start is None and video_end is None:
        return 0, total_frames - 1, total_frames

    max_duration = total_frames / video_fps
    # Process start frame
    if video_start is not None:
        video_start_clamped = max(0.0, min(video_start, max_duration))
        start_frame = math.ceil(video_start_clamped * video_fps)
    else:
        start_frame = 0
    # Process end frame
    if video_end is not None:
        video_end_clamped = max(0.0, min(video_end, max_duration))
        end_frame = math.floor(video_end_clamped * video_fps)
        end_frame = min(end_frame, total_frames - 1)
    else:
        end_frame = total_frames - 1

    # Validate frame order
    if start_frame >= end_frame:
        raise ValueError(
            f"Invalid time range: Start frame {start_frame} (at {video_start_clamped if video_start is not None else 0}s) "
            f"exceeds end frame {end_frame} (at {video_end_clamped if video_end is not None else max_duration}s). "
            f"Video duration: {max_duration:.2f}s ({total_frames} frames @ {video_fps}fps)"
        )

    logger.info(f"calculate video frame range: {start_frame=}, {end_frame=}, {total_frames=} from {video_start=}, {video_end=}, {video_fps=:.3f}")
    return start_frame, end_frame, end_frame - start_frame + 1


def _read_video_decord(
    ele: Dict[str, Any],
) -> Tuple[torch.Tensor, float]:
    """read video using decord.VideoReader

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    import decord
    video_path = ele["video"]
    st = time.time()
    vr = decord.VideoReader(video_path)
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    start_frame, end_frame, total_frames = calculate_video_frame_range(
        ele,
        total_frames,
        video_fps,
    )
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
    video = vr.get_batch(idx).asnumpy()
    video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
    logger.info(f"decord:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps

    video_metadata = dict(
        fps=video_fps,
        frames_indices=idx,
        total_num_frames=total_frames,
        video_backend="decord",
    )
    return video, video_metadata, sample_fps


def is_torchcodec_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("torchcodec") is not None


def _read_video_torchcodec(
    ele: Dict[str, Any],
) -> Tuple[torch.Tensor, float]:
    """read video using torchcodec.decoders.VideoDecoder

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    from torchcodec.decoders import VideoDecoder
    TORCHCODEC_NUM_THREADS = int(os.environ.get('TORCHCODEC_NUM_THREADS', 8))
    logger.info(f"set TORCHCODEC_NUM_THREADS: {TORCHCODEC_NUM_THREADS}")
    video_path = ele["video"]
    st = time.time()
    decoder = VideoDecoder(video_path, num_ffmpeg_threads=TORCHCODEC_NUM_THREADS)
    video_fps = decoder.metadata.average_fps
    total_frames = decoder.metadata.num_frames
    start_frame, end_frame, total_frames = calculate_video_frame_range(
        ele,
        total_frames,
        video_fps,
    )
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    video = decoder.get_frames_at(indices=idx).data
    logger.info(f"torchcodec:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")

    video_metadata = dict(
        fps=video_fps,
        frames_indices=idx,
        total_num_frames=total_frames,
        video_backend="torchcodec",
    )
    return video, video_metadata, sample_fps


VIDEO_READER_BACKENDS = {
    "decord": _read_video_decord,
    "torchvision": _read_video_torchvision,
    "torchcodec": _read_video_torchcodec,
}

FORCE_QWENVL_VIDEO_READER = os.getenv("FORCE_QWENVL_VIDEO_READER", None)


@lru_cache(maxsize=1)
def get_video_reader_backend() -> str:
    if FORCE_QWENVL_VIDEO_READER is not None:
        video_reader_backend = FORCE_QWENVL_VIDEO_READER
    elif is_torchcodec_available():
        video_reader_backend = "torchcodec"
    elif is_decord_available():
        video_reader_backend = "decord"
    else:
        video_reader_backend = "torchvision"
    print(f"qwen-vl-utils using {video_reader_backend} to read video.", file=sys.stderr)
    return video_reader_backend


def _get_cache_key(video_path: str, ele: Dict[str, Any]) -> str:
    """Generate a unique cache key based on video path and processing parameters."""
    video_basename = os.path.basename(video_path).split(".")[0]
    
    # Include relevant processing parameters in the key
    params = []
    for key in ["video_start", "video_end", "fps", "nframes", "min_frames", "max_frames", 
                "min_pixels", "max_pixels", "resized_height", "resized_width"]:
        if key in ele:
            params.append(f"{key}={ele[key]}")
    
    if params:
        param_str = "_".join(params)
        # Hash long parameter strings to keep filename manageable
        if len(param_str) > 50:
            import hashlib
            param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
            return f"{video_basename}_{param_hash}"
        return f"{video_basename}_{param_str}"
    
    return video_basename


def _validate_cached_data(cached_data: Dict[str, Any]) -> bool:
    """
    Validate that cached data contains all required keys and valid data.
    Handles both old and new cache formats for backward compatibility.
    
    Args:
        cached_data: Dictionary containing cached video data
        
    Returns:
        bool: True if cache is valid, False otherwise
    """
    if not isinstance(cached_data, dict):
        logger.warning("Cached data is not a dictionary")
        return False
    
    # Check if this is the new format (preferred)
    if "video" in cached_data and "video_metadata" in cached_data and "sample_fps" in cached_data:
        # Validate new format
        video = cached_data.get("video")
        if not isinstance(video, torch.Tensor):
            logger.warning("Cached video is not a torch.Tensor")
            return False
        
        if video.dim() != 4:  # Should be (T, C, H, W)
            logger.warning(f"Cached video has invalid dimensions: {video.shape}")
            return False
        
        # Validate video_metadata
        video_metadata = cached_data.get("video_metadata")
        if not isinstance(video_metadata, dict):
            logger.warning("Cached video_metadata is not a dictionary")
            return False
        
        # Check for required metadata keys
        metadata_keys = ["fps", "frames_indices", "total_num_frames"]
        for key in metadata_keys:
            if key not in video_metadata:
                logger.warning(f"Cached video_metadata missing key: {key}")
                return False
        
        # Validate sample_fps
        sample_fps = cached_data.get("sample_fps")
        if not isinstance(sample_fps, (int, float)) or sample_fps <= 0:
            logger.warning(f"Cached sample_fps is invalid: {sample_fps}")
            return False
        
        return True
    
    # Check if this is the old format (legacy support)
    elif "video_inputs" in cached_data and "video_metadata" in cached_data:
        logger.info("Detected old cache format, will migrate to new format")
        return True  # Allow old format to pass validation, will be migrated
    
    else:
        logger.warning("Cached data format is unrecognized (neither new nor old format)")
        return False


def _migrate_old_cache_format(old_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Migrate old cache format to new format.
    
    Args:
        old_data: Dictionary in old cache format
        
    Returns:
        Dictionary in new cache format, or None if migration fails
    """
    try:
        # Extract video data from old format
        video_inputs = old_data.get("video_inputs")
        video_metadata = old_data.get("video_metadata", {})
        
        if not video_inputs or len(video_inputs) == 0:
            logger.warning("Old cache format has no video_inputs")
            return None
        
        # For single video, take the first one
        video_tensor = video_inputs[0] if isinstance(video_inputs, list) else video_inputs
        
        if not isinstance(video_tensor, torch.Tensor):
            logger.warning("Old cache format video_inputs is not a tensor")
            return None
        
        # Extract sample_fps from video_kwargs if available
        video_kwargs = old_data.get("video_kwargs", {})
        sample_fps = video_kwargs.get("fps", [2.0])[0] if "fps" in video_kwargs else 2.0
        
        # Create new format data
        new_data = {
            "video": video_tensor,
            "video_metadata": video_metadata,
            "sample_fps": sample_fps,
        }
        
        logger.info("Successfully migrated old cache format to new format")
        return new_data
        
    except Exception as e:
        logger.warning(f"Failed to migrate old cache format: {e}")
        return None


def _load_cached_video(cache_filepath: str) -> Optional[Dict[str, Any]]:
    """Load cached video data from disk with validation and migration."""
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
        
        # Check if this is old format and needs migration
        if "video_inputs" in data and "video" not in data:
            logger.info(f"Migrating old cache format: {cache_filepath}")
            migrated_data = _migrate_old_cache_format(data)
            if migrated_data is not None:
                # Save migrated data back to cache
                try:
                    _save_cached_video(cache_filepath, 
                                     migrated_data["video"], 
                                     migrated_data["video_metadata"], 
                                     migrated_data["sample_fps"])
                    logger.info(f"Successfully migrated and saved cache: {cache_filepath}")
                    return migrated_data
                except Exception as e:
                    logger.warning(f"Failed to save migrated cache: {e}")
                    return migrated_data
            else:
                logger.warning(f"Failed to migrate old cache format: {cache_filepath}")
                # Delete old cache file
                try:
                    os.remove(cache_filepath)
                    logger.info(f"Removed unmigratable cache file: {cache_filepath}")
                except Exception as e:
                    logger.warning(f"Failed to remove unmigratable cache file: {e}")
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


def _save_cached_video(cache_filepath: str, video: torch.Tensor, video_metadata: Dict[str, Any], sample_fps: float):
    """Save processed video data to cache."""
    try:
        save_data = {
            "video": video,
            "video_metadata": video_metadata,
            "sample_fps": sample_fps,
        }
        
        # Create temporary file first, then rename (atomic operation on most systems)
        temp_filepath = cache_filepath + ".tmp"
        with open(temp_filepath, "wb") as f:
            pickle.dump(save_data, f)
        
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


def fetch_video(ele: Dict[str, Any], image_patch_size: int = 14, return_video_sample_fps: bool = False,
                return_video_metadata: bool = True, use_cache: bool = True) -> Union[torch.Tensor, List[Image.Image]]:
    image_factor = image_patch_size * SPATIAL_MERGE_SIZE
    VIDEO_FRAME_MIN_PIXELS = VIDEO_MIN_TOKEN_NUM * image_factor * image_factor
    VIDEO_FRAME_MAX_PIXELS = VIDEO_MAX_TOKEN_NUM * image_factor * image_factor
    
    video = None
    video_metadata = None
    sample_fps = None
    
    # Handle string video path (file/url)
    if isinstance(ele["video"], str):
        # Check cache first if enabled
        cache_key = _get_cache_key(ele["video"], ele)
        cache_filepath = os.path.join(QWEN_VL_VIDEO_CACHE_DIR, f"{cache_key}.pkl")
        
        if use_cache and os.path.exists(cache_filepath):
            cached_data = _load_cached_video(cache_filepath)
            if cached_data is not None:
                video = cached_data["video"]
                video_metadata = cached_data["video_metadata"]
                sample_fps = cached_data["sample_fps"]
        
        # Process video if not loaded from cache
        if video is None:
            video_reader_backend = get_video_reader_backend()
            try:
                video, video_metadata, sample_fps = VIDEO_READER_BACKENDS[video_reader_backend](ele)
            except Exception as e:
                logger.warning(
                    f"video_reader_backend {video_reader_backend} error, attempting fallback(s). msg: {e}"
                )
                # Prefer torchcodec (fast and robust) if available, then torchvision as last resort
                if is_torchcodec_available():
                    try:
                        logger.warning("Trying torchcodec as fallback backend...")
                        video, video_metadata, sample_fps = VIDEO_READER_BACKENDS["torchcodec"](ele)
                    except Exception as e_tc:
                        logger.warning(
                            f"torchcodec fallback failed, trying torchvision. msg: {e_tc}"
                        )
                        try:
                            video, video_metadata, sample_fps = VIDEO_READER_BACKENDS["torchvision"](ele)
                        except Exception as e_tv:
                            raise RuntimeError(
                                "All video backends failed (primary, torchcodec, torchvision). "
                                "Consider setting FORCE_QWENVL_VIDEO_READER=torchcodec if installed, "
                                f"or investigate ffmpeg/torchvision setup. Last error: {e_tv}"
                            )
                else:
                    logger.warning("torchcodec not available, falling back to torchvision...")
                    try:
                        video, video_metadata, sample_fps = VIDEO_READER_BACKENDS["torchvision"](ele)
                    except Exception as e_tv:
                        raise RuntimeError(
                            "Video decoding failed for both primary backend and torchvision. "
                            "Install torchcodec or decord, or set FORCE_QWENVL_VIDEO_READER accordingly. "
                            f"Last error: {e_tv}"
                        )
            
            # Perform resizing
            nframes, _, height, width = video.shape
            min_pixels = ele.get("min_pixels", VIDEO_FRAME_MIN_PIXELS)
            total_pixels = ele.get("total_pixels", MODEL_SEQ_LEN * image_factor * image_factor * 0.9)
            max_pixels = max(min(VIDEO_FRAME_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
            max_pixels_supposed = ele.get("max_pixels", max_pixels)
            if max_pixels_supposed > max_pixels:
                logger.warning(f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}].")
            max_pixels = min(max_pixels_supposed, max_pixels)
            
            if "resized_height" in ele and "resized_width" in ele:
                resized_height, resized_width = smart_resize(
                    ele["resized_height"],
                    ele["resized_width"],
                    factor=image_factor,
                )
            else:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=image_factor,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                )
            
            video = transforms.functional.resize(
                video,
                [resized_height, resized_width],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ).float()
            
            # Save to cache if enabled
            if use_cache:
                _save_cached_video(cache_filepath, video, video_metadata, sample_fps)
    
    # Handle list/tuple of frames
    else:
        assert isinstance(ele["video"], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        
        # use ThreadPoolExecutor to parallel process frames
        max_workers = min(MAX_NUM_WORKERS_FETCH_VIDEO, len(ele["video"]))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(fetch_image, {"image": video_element, **process_info}, image_factor)
                for video_element in ele["video"]
            ]
            image_list = [future.result() for future in futures]

        nframes = ceil_by_factor(len(image_list), FRAME_FACTOR)
        if len(image_list) < nframes:
            image_list.extend([image_list[-1]] * (nframes - len(image_list)))

        sample_fps = ele.get("sample_fps", 2.0)
        video = torch.stack([
            torch.from_numpy(np.array(image).transpose(2, 0, 1))
            for image in image_list
        ])

        # fake video metadata
        raw_fps = process_info.pop("raw_fps", sample_fps)
        video_metadata = dict(
            fps=raw_fps,
            frames_indices=[i for i in range(len(video))],
            total_num_frames=(nframes / sample_fps) * raw_fps,
        )

    # Prepare return value based on flags
    final_video = (video, video_metadata) if return_video_metadata else video
    if return_video_sample_fps:
        return final_video, sample_fps
    return final_video


def extract_vision_info(conversations: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]) -> List[Dict[str, Any]]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                        "image" in ele
                        or "image_url" in ele
                        or "video" in ele
                        or ele.get("type", "text") in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)
    return vision_infos


def process_vision_info(
    conversations: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
    return_video_kwargs: bool = False,
    return_video_metadata: bool = False,
    image_patch_size: int = 14,
    use_cache: bool = False,
) -> Tuple[Optional[List[Image.Image]], Optional[List[Union[torch.Tensor, List[Image.Image]]]], Optional[Dict[str, Any]]]:

    vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []
    video_paths = []
    
    def _is_timeout_error(err: Exception) -> bool:
        msg = str(err).lower()
        # Common timeout substrings from torchcodec/requests/ffmpeg
        return (
            "timed out" in msg
            or "timeout" in msg
            or "operation timed out" in msg
        )

    video_timeout_occurred = False

    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info, image_patch_size=image_patch_size))
        elif "video" in vision_info:
            # fetch_video now handles caching internally
            try:
                video_input, video_sample_fps = fetch_video(
                    vision_info,
                    return_video_sample_fps=True,
                    image_patch_size=image_patch_size,
                    return_video_metadata=return_video_metadata,
                    use_cache=use_cache,
                )
                video_paths.append(vision_info["video"])
                video_sample_fps_list.append(video_sample_fps)
                video_inputs.append(video_input)
            except Exception as e:
                if _is_timeout_error(e):
                    logger.warning(
                        f"video_reader_backend timeout encountered for {vision_info.get('video')}, returning empty video inputs list. msg: {e}"
                    )
                    video_timeout_occurred = True
                    # Clear any partially collected video info to return an empty list per requirement
                    video_inputs = []
                    video_sample_fps_list = []
                    video_paths = []
                    # Stop further video processing
                    break
                # Re-raise non-timeout errors
                raise
        else:
            raise ValueError("image, image_url or video should in content.")
    
    if len(image_inputs) == 0:
        image_inputs = None
    # If a timeout occurred, return empty list for video_inputs as requested
    if not video_timeout_occurred and len(video_inputs) == 0:
        video_inputs = None

    video_kwargs = {'do_sample_frames': False}
    if not return_video_metadata: # BC for qwen2.5vl
        video_kwargs.update({'fps': video_sample_fps_list, 'video_paths': video_paths})

    if return_video_kwargs:
        return image_inputs, video_inputs, video_kwargs
    return image_inputs, video_inputs