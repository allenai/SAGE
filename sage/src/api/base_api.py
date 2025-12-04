from typing import List, Optional, Union, Dict, Any
import torch
from PIL import Image
import logging
import os
import gc
import random
from abc import ABC
logger = logging.getLogger(__name__)

from sage.utils.utils import (
    VISUAL_TEMPORAL_GROUNDING_PROMPT,
)
from sage.src.functions.utils.extract import extract_subclip
from sage.src.functions.utils.temporal import timestamp_to_seconds, get_video_duration, fix_timestamp, seconds_to_timestamp
from sage.utils.json_parser import clean_json
from vllm import LLM
from transformers import AutoProcessor
import base64
from sage.src.models.molmo2 import Molmo2ForConditionalGeneration as VideoMolmo2ForConditionalGeneration
from sage.src.api.timestamp_reasoning_cache import (
    get_cached_timestamp_response, 
    set_cached_timestamp_response,
    get_cached_reasoning_response,
    set_cached_reasoning_response
)
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.0))
print(f"TEMPERATURE: {TEMPERATURE}")

USE_VLLM = os.environ.get("USE_VLLM", "True").lower() == "true"
USE_GEMINI_AS_TOOL = os.environ.get("USE_GEMINI_AS_TOOL", "False").lower() == "true"
TOOL_CALL_MODEL = os.environ.get("TOOL_CALL_MODEL")
MAX_GEN_TOKENS = int(os.environ.get("MAX_GEN_TOKENS", "32768"))
VLLM_CLIENT_URL = os.environ.get("VLLM_CLIENT_URL", "None")

def get_vllm_urls() -> List[str]:
    if VLLM_CLIENT_URL == "None" or not VLLM_CLIENT_URL:
        return VLLM_CLIENT_URL
    
    urls = [url.strip() for url in VLLM_CLIENT_URL.split(",") if url.strip()]
    if not urls:
        return VLLM_CLIENT_URL

    return urls

def prepare_message_for_vllm(content_messages):
    """
    The frame extraction logic for videos in `vLLM` differs from that of `qwen_vl_utils`.
    Here, we utilize `qwen_vl_utils` to extract video frames, with the `media_typ`e of the video explicitly set to `video/jpeg`.
    By doing so, vLLM will no longer attempt to extract frames from the input base64-encoded images.
    """
    vllm_messages, fps_list = [], []
    for message in content_messages:
        message_content_list = message["content"]
        if not isinstance(message_content_list, list):
            vllm_messages.append(message)
            continue

        new_content_list = []
        for part_message in message_content_list:
            if 'video' in part_message:
                # Get the video path from the part_message
                video_path = part_message['video']
                
                # Read the video file and encode it as base64
                with open(video_path, "rb") as video_file:
                    video_content = video_file.read()
                    video_base64 = base64.b64encode(video_content).decode("utf-8")
                
                # Create the proper video_url format
                part_message = {
                    "type": "video_url",
                    "video_url": {"url": f"data:video/mp4;base64,{video_base64}"},

                }
            new_content_list.append(part_message)
        message["content"] = new_content_list
        vllm_messages.append(message)
    return vllm_messages, {'fps': fps_list}

class BaseAPI(ABC):
    def __init__(
        self,
        model_name: str = None,
        vllm_engine: object = None,
        processor: AutoProcessor = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = MAX_GEN_TOKENS,
        temperature: float = 0.7,
        top_p: float = 0.9,
        is_rl_train_mode: bool = False,
        use_video: bool = True,
        tool_call_clients: List[object] = None,
        model_class: Any = None,
        is_molmo2: bool = False,
    ):
        """
        Initialize the Base API model.

        Args:
            model_name: Name of the model on Hugging Face Hub
            device: Device to run the model on ('cuda' or 'cpu')
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
        """
        self.device = device
        self.max_length = max_length
        self.temperature = TEMPERATURE if TEMPERATURE is not None else temperature
        self.top_p = top_p
        self.vllm_engine = vllm_engine
        self.processor = processor
        self.is_rl_train_mode = is_rl_train_mode
        self.use_video = use_video
        self.generation_kwargs = {
                "n": 1,  # vLLM on each GPU generates only 1 in colocate mode
                "repetition_penalty": 1.0,
                "temperature": self.temperature,
                "top_p": 0.9,
                "top_k": -1,
                "min_p": 0.0,
                "max_tokens": self.max_length,
                "guided_decoding": None,
            }
        
        if not is_rl_train_mode and vllm_engine is None:
            if ":" in model_name:
                model_path = model_name.split(":")[-1]
            if USE_VLLM:
                self.vllm_engine = LLM(
                        model=model_path,
                        tensor_parallel_size=len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")),
                        gpu_memory_utilization=0.8,
                        max_num_seqs=1,
                        max_model_len=self.max_length,
                        seed=0,
                        max_num_batched_tokens=self.max_length,
                        trust_remote_code=True
                    )
                self.vllm_engine.reset_prefix_cache()
            else:
                self.model = model_class.from_pretrained(
                    model_path, device_map="auto", trust_remote_code=True, 
                    low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, 
                    attn_implementation="flash_attention_2" if not is_molmo2 else "sdpa"
                ).eval()
            if not USE_GEMINI_AS_TOOL:
                print(f"TOOL_CALL_MODEL: {TOOL_CALL_MODEL}")
                
                if TOOL_CALL_MODEL == "None":
                    self.tool_call_model = None
                else:
                    self.tool_call_model = TOOL_CALL_MODEL
                    from openai import OpenAI
                    self.tool_call_clients = [
                        OpenAI(
                            base_url=url, 
                            api_key="not-needed"
                        ) 
                        for url in get_vllm_urls()
                    ]
        elif tool_call_clients is not None:
            self.tool_call_clients = tool_call_clients
            if self.tool_call_clients is not None:
                self.tool_call_model = TOOL_CALL_MODEL
            else:
                self.tool_call_model = None
        else:
            if not USE_GEMINI_AS_TOOL:
                print(f"TOOL_CALL_MODEL: {TOOL_CALL_MODEL}")
                
                if TOOL_CALL_MODEL == "None":
                    self.tool_call_model = None
                else:
                    self.tool_call_model = TOOL_CALL_MODEL
                    from openai import OpenAI
                    self.tool_call_clients = [
                        OpenAI(
                            base_url=url, 
                            api_key="not-needed"
                        ) 
                        for url in get_vllm_urls()
                    ]

    def cleanup(self):
      if self.vllm_engine is not None:
          print("Deleting vllm engine...")
          del self.vllm_engine
          self.vllm_engine = None
          gc.collect()

    def _prepare_input(
        self, 
        prompt: str, 
        media: Optional[Union[str, Image.Image, List[Image.Image]]] = [], 
        media_type: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> Any:
        raise NotImplementedError("Implement this in the child class")
    
    def _prepare_vllm_messages(
        self, 
        prompt: str, 
        media: Optional[Union[str, Image.Image, List[Image.Image]]] = [], 
        media_type: Optional[str] = None, 
    ) -> Any:
        """
        Prepare messages for the chat API.

        Args:
            media: Media input (path to image/video, PIL Image, or list of PIL Images)
            prompt: Text prompt

        Returns:
            Formatted query for the model
        """
        if media_type is not None:
            assert media_type in ["image", "video"], f"Invalid media type: {media_type}"
        
        messages = [
            {
                "role": "system",
                "content": "Respond in a concise and direct manner to the user's question."
            }
        ]
        
        if media_type == "video":
            media = [media[0]]
        
        if media_type is not None:
            if media_type == "image":
                content_parts = []
                for media_path in media:
                    with open(media_path, "rb") as f:
                        encoded_image = base64.b64encode(f.read())
                    encoded_image_text = encoded_image.decode("utf-8")
                    base64_qwen = f"data:image;base64,{encoded_image_text}"
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": base64_qwen
                        },
                    })
                content_parts.append({"type": "text", "text": prompt})
                messages.append({
                    "role": "user",
                    "content": content_parts
                })
            else:
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": media_type,
                        media_type: media[0]
                    }, {
                        "type": "text", 
                        "text": prompt
                    }]
                })
                messages, _ = prepare_message_for_vllm(messages)
        else:
            if not self.use_video:
                print("Not using video")
            messages.append({
                "role": "user",
                "content": prompt
            })

        return messages
    
    def get_response(
        self,
        prompt: str,
        media: Optional[Union[str, Image.Image, List[Image.Image]]] = None,
        media_type: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        return_ids: bool = False,
        temperature: float = None,
        **kwargs,
    ) -> str:
        raise NotImplementedError("Implement this in the child class")
    
    def identify_timestamps_visually(self, video_path: str, event: str, timestamp_start: str, timestamp_end: str, **kwargs) -> str:

        if not video_path.endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
            raise ValueError(f"Video file does not have a valid extension: {video_path}")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file does not exist: {video_path}")
        
        # Check cache first
        model_name = getattr(self, 'model_name', None) or getattr(self, 'tool_call_model', None)
        cached_response = get_cached_timestamp_response(video_path, event, timestamp_start, timestamp_end, model_name)
        if cached_response is not None:
            print(f"Using cached timestamp identification for event: {event}")
            return cached_response
        
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
        
        if self.tool_call_model is None:
            with torch.no_grad():
                response = self.get_response(
                    prompt,
                    media=[video_path],
                    media_type="video",
                    temperature=0.0,
                    **kwargs,
                )
        else:
            messages = self._prepare_vllm_messages(prompt, media=[video_path], media_type="video")
            response = random.choice(self.tool_call_clients).chat.completions.create(
                    model=self.tool_call_model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=8192,
                )
            response = response.choices[0].message.content
        if "<json>" in response:
            while "<json>" in response:
                response = response.split("<json>")[1].strip().split("</json>")[0].strip()
            try:
                response = clean_json(response)
            except Exception as e:
                print("Error in response: ", e)
                raise e
        
        if isinstance(response, dict):
            timestamps = response.get("timestamps", {})
            if timestamps.get("start", None) is not None and timestamps.get("end", None) is not None and timestamps.get("start") != timestamps.get("end"):
                if timestamp_to_seconds(timestamps["start"]) < timestamp_start_seconds:
                    timestamps["start"] = seconds_to_timestamp(timestamp_start_seconds + timestamp_to_seconds(timestamps["start"]))
                    timestamps["end"] = seconds_to_timestamp(timestamp_start_seconds + timestamp_to_seconds(timestamps["end"]))
            response["timestamps"] = timestamps
        
        # Cache the response
        set_cached_timestamp_response(video_path, event, timestamp_start, timestamp_end, response, model_name)
        return response
    
    def perform_reasoning(self, query: str, media_paths: List[str], **kwargs) -> Dict[str, Any]:
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
        
        # Check cache first
        model_name = getattr(self, 'model_name', None) or getattr(self, 'tool_call_model', None)
        cached_response = get_cached_reasoning_response(query, media_paths or [], media_type, model_name)
        if cached_response is not None:
            print(f"Using cached reasoning response for query: {query[:50]}...")
            return cached_response
        
        if self.tool_call_model is None:
            with torch.no_grad():
                response = self.get_response(query, media=media_paths, media_type=media_type, temperature=0.0, **kwargs)
        else:
            messages = self._prepare_vllm_messages(query, media=media_paths, media_type=media_type)
            response = random.choice(self.tool_call_clients).chat.completions.create(
                model=self.tool_call_model,
                messages=messages,
                temperature=0.0,
                max_tokens=8192,
            )
            response = response.choices[0].message.content
        
        result = {"answer": response}
        
        # Cache the response
        set_cached_reasoning_response(query, media_paths or [], media_type, result, model_name)
        return result

if __name__ == "__main__":
    # This is a base class - use specific implementations like QwenVL
    print("BaseAPI is a base class. Use specific implementations like QwenVL.")