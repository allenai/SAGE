import os
import requests
import json
from typing import Optional, List, Dict, Any, Tuple
from sage.utils.utils import (
    CONTEXT_VLM_PROMPT,
    SAGE_CONTEXT_VLM_PROMPT,
    SAGE_ITERATIVE_REASONER_PROMPT,
)
from sage.src.functions.utils.extract import extract_subclip, extract_frames
from sage.src.api import Gemini, GPT, QwenVL, Molmo2, Qwen3_VL
from sage.src.functions.utils.temporal import seconds_to_timestamp
from google.genai import types
import time
import cv2
from tqdm import tqdm
from sage.utils.json_parser import clean_json
from sage.src.functions.utils.utils import get_functions_from_folder, format_tool_descs

class ContextVLM:
    """
    A class to handle video content understanding using Vision Language Models like GPT-4o or Gemini.
    """

    def __init__(self, 
            api_type: str = None, use_vertexai: bool = False, 
            vllm_engine: object = None, processor: object = None, 
            drop_tool_call_files: list = [],
            is_rl_train_mode: bool = False,
            use_video: bool = True,
            tool_call_clients: List[object] = None,
        ):
        """
        Initialize the Context VLM.

        Args:
            model: Model name to use (e.g., "gpt-4o", "gemini-pro-vision")
            api_type: Type of API to use ("gemini" or "gpt")
        """
        self.api_type = api_type.lower()
        self.use_video = use_video
        if "qwen3" in self.api_type:
            self.client = Qwen3_VL(
                model_name=api_type,
                vllm_engine=vllm_engine,
                processor=processor,
                is_rl_train_mode=is_rl_train_mode,
                use_video=use_video,
                tool_call_clients=tool_call_clients,
            )
        elif "qwen" in self.api_type:
            self.client = QwenVL(
                model_name=api_type,
                vllm_engine=vllm_engine,
                processor=processor,
                is_rl_train_mode=is_rl_train_mode,
                use_video=use_video,
                tool_call_clients=tool_call_clients,
            )
        elif "molmo2" in self.api_type:
            self.client = Molmo2(
                model_name=api_type,
                vllm_engine=vllm_engine,
                processor=processor,
                is_rl_train_mode=is_rl_train_mode,
                use_video=use_video,
                tool_call_clients=tool_call_clients,
            )
        elif "gpt" in self.api_type:
            self.client = GPT()
        else:
            self.use_vertexai = use_vertexai
            self.client = Gemini(use_vertexai=use_vertexai)
        
        self.tools, _ = get_functions_from_folder("sage/src/functions/tools", drop_tool_call_files=drop_tool_call_files)

    def get_token_count(self, video_path: str) -> int:
        if "gemini" in self.api_type and not self.use_vertexai:
            return self.client.count_tokens(video_path)
        else:
            return 0
    
    def _retry_with_exponential_backoff(self, func, *args, **kwargs):
        if "gemini" in self.api_type:
            max_attempts = 5
        else:
            max_attempts = 1
        last_exception = None
        for attempt in range(max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if "gemini" in self.api_type:
                    print(f"Error: {e}, retrying in {2 ** attempt} seconds...")
                    time.sleep(2**attempt)
                else:
                    print(f"Error: {e}, retrying now")
        
        # If all retries failed, re-raise the last exception
        if last_exception is not None:
            raise last_exception

    def delete_client_files(self):
        def _delete_client_files():
            for f in tqdm(self.client.client.files.list(), desc="Deleting files"):
                try:
                    self.client.client.files.delete(name=f.name)
                except Exception as e:
                    if "403 PERMISSION_DENIED" in str(e):
                        pass
                    else:
                        raise e

        if "gemini" in self.api_type and not self.use_vertexai:
            self._retry_with_exponential_backoff(_delete_client_files)
        else:
            pass

    def process_video(self, video_path: str, full_video: bool = False) -> str:
        assert os.path.exists(video_path), "Video file does not exist: {}".format(video_path)
        assert video_path.endswith(
            (".mp4", ".mpeg", ".mov", ".avi", ".flv", ".mpg", ".webm", ".wmv", ".3gp")
        ), "Unsupported video format: {}".format(video_path.split(".")[-1])

        if not full_video:
            video_path = extract_subclip(
                video_path,
                0,
                min(self.get_video_duration(video_path), self.context_video_duration),
            )
        return video_path

    def answer(
        self,
        video_path: str,
        query: str,
        model_name: str = "gemini-2.5-flash",
        sample_frames: bool = False,
        num_sampled_frames: int = 50,
        **kwargs,
    ) -> str:
        model_name = model_name.lower()
        if "gemini" in model_name and not sample_frames:
            video_path = self.process_video(video_path, full_video=True)
            media_paths = [video_path]
            media_type = "video"   
        elif "gpt" in model_name:
            media_paths = extract_frames(
                video_path,
                0,
                self.get_video_duration(video_path),
                num_frames=min(num_sampled_frames, int(self.get_video_duration(video_path))),
            )
            media_type = "image"
        else:
            media_paths = [video_path]
            media_type = "video"

        response = self.run_inference(query, model_name, media_paths, media_type, **kwargs)

        return response["vlm_response"]

    def run_inference(
        self,
        query: str,
        model_name: str = "gemini:gemini-2.5-flash",
        media_paths: List[str] = None,
        media_type: str = None,
        parse_json: bool = False,
        return_ids: bool = False,
        temperature: float = None,
        **kwargs,
    ) -> str:
        model_name = model_name.lower()
        api_type, model_name = model_name.split(":")
        completion_ids = []
        prompt_ids = []
        attention_mask = []
        if "qwen" in model_name or "molmo2" in model_name:
            system_prompt = SAGE_CONTEXT_VLM_PROMPT
            response = self.client.get_response(
                prompt=query,
                media=media_paths,
                media_type=media_type,
                system_prompt=system_prompt.replace("<<<tools>>>", format_tool_descs(self.tools)),
                return_ids=return_ids,
                temperature=temperature,
                **kwargs,
            )
            
            if return_ids:
                response, completion_ids, prompt_ids, attention_mask = response
        elif "gemini" in model_name:
            response = self.client.get_response(
                model_name=model_name,
                query=query,
                media_paths=media_paths,
                media_type=media_type,
                temperature=temperature,
            )["answer"]
        else:
            response = self.client.get_response(prompt=query, image_urls=media_paths, model=model_name)[0]

        if response is None:
            response = ""

        if parse_json:
            if "```json" in response:
                try:
                    response = response.split("```json")[1].split("```")[0]
                    response = clean_json(response)
                except Exception as e:
                        if return_ids:
                            response = response
                        else:
                            raise e
            elif "<json>" in response:
                try:
                    while "<json>" in response:
                        response = response.split("<json>")[1].strip().split("</json>")[0].strip()
                    response = clean_json(response)
                except Exception as e:
                        if return_ids:
                            response = response
                        else:
                            raise e
            else:
                if return_ids:
                    response = response
                else:
                    raise ValueError("Invalid response format")
        
        result_dict = {
            "vlm_response": response,
            "completion_ids": completion_ids,
            "prompt_ids": prompt_ids,
            "attention_mask": attention_mask,
        }

        return result_dict

    def get_context_vlm_prompt(
        self,
        query: str,
        video_path: str,
        tools: List[types.Tool] = None,
        sample_frames: bool = False,
        num_sampled_frames: int = 128,
    ) -> str:

        if not self.use_video:
            timestamp_format = "video duration not available"
            video_path = "video not provided"
        else:
            video_duration = self.get_video_duration(video_path)
            timestamp_format = seconds_to_timestamp(video_duration, in_hr=True)
        media_paths = []
        media_type = None
        if "qwen" in self.api_type or "molmo2" in self.api_type:
            query = query + "\n" + f"Video Path: {video_path} of duration {timestamp_format}."
            if self.use_video:
                media_paths = [video_path]
                media_type = "video"
            return query, media_paths, media_type
        else:
            if self.use_video:
                media_paths = extract_frames(
                    video_path,
                    0,
                    self.get_video_duration(video_path),
                    num_frames=min(num_sampled_frames, int(self.get_video_duration(video_path))),
                )
                media_type = "image"
            query = (
                SAGE_CONTEXT_VLM_PROMPT.replace("<<<query>>>", query)
                .replace("<<<tools>>>", format_tool_descs(tools))
                .replace("<<<video_path>>>", video_path)
                .replace("<<<video_duration>>>", str(video_duration))
                .replace(
                    "<<<timestamp_format>>>",
                    timestamp_format,
                )
            )

        return query, media_paths, media_type

    def analyze(
        self,
        video_path: str,
        query: str,
        model_name: str = "gemini:gemini-2.5-flash",
        tools: List[types.Tool] = None,
        sample_frames: bool = False,
        use_transcript: bool = False,
        num_sampled_frames: int = 128,
        return_ids: bool = False,
        temperature: float = None,
        **kwargs,
    ) -> str:
    
        query, media_paths, media_type = self.get_context_vlm_prompt(
            query, video_path, tools, sample_frames, num_sampled_frames
        )

        if return_ids:
            result_dict = self.run_inference(
                query,
                model_name,
                media_paths,
                media_type,
                parse_json=True,
                return_ids=return_ids,
                temperature=temperature,
                **kwargs,
            )
        else:
            result_dict = self._retry_with_exponential_backoff(
                self.run_inference,
                query,
                model_name,
                media_paths,
                media_type,
                parse_json=True,
                return_ids=return_ids,
                temperature=temperature,
                **kwargs,
            )
            
        # result_dict["inputs"] = {
        #     "video_path": video_path,
        #     "query": query,
        # }
        return result_dict

    def get_video_duration(self, video_path: str) -> float:
        if not video_path.endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
            raise ValueError(f"Video file does not have a valid extension: {video_path}")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file does not exist: {video_path}")
        cap = cv2.VideoCapture(video_path)
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
