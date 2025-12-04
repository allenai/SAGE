import os
import requests
import json
import logging
from typing import Optional, List, Dict, Any, Tuple
from google import genai
from sage.utils.utils import (
    GCP_PROJECT_ID,
    CONTEXT_VLM_PROMPT,
    WEB_SEARCH,
    API_KEYS,
)
from sage.src.functions.utils.utils import upload_to_gcp_bucket
from sage.src.api.gemini_cache import (
    get_cached_response as gemini_get_cached_response,
    set_cached_response as gemini_set_cached_response,
    clear_cache as gemini_clear_cache,
    get_cache_stats as gemini_get_cache_stats,
)
from google.genai import types
from google.genai.types import HttpOptions, Part
import time
from tqdm import tqdm
import cv2
from concurrent.futures import ThreadPoolExecutor

# Suppress HTTP request logs from Google API client
logging.basicConfig(level=logging.WARNING)
logging.getLogger('google.api_core.http').setLevel(logging.WARNING)
logging.getLogger('google.auth.transport.requests').setLevel(logging.WARNING)
logging.getLogger('google.api_core').setLevel(logging.WARNING)
logging.getLogger('google.auth').setLevel(logging.WARNING)
logging.getLogger('google').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('google.genai').setLevel(logging.WARNING)
logging.getLogger('google.generativeai').setLevel(logging.WARNING)


TEMPERATURE=float(os.environ.get("TEMPERATURE", 0.7))
print(f"TEMPERATURE: {TEMPERATURE}")

class Gemini:
    def __init__(self, use_vertexai: bool = False, enable_cache: bool = True):
        self.use_vertexai = use_vertexai
        self.enable_cache = enable_cache
        if use_vertexai:
            self.client = genai.Client(vertexai=True, location="us-central1", project=GCP_PROJECT_ID)
        else:
            self.client = genai.Client(api_key=API_KEYS.get("gemini"))

    def clear_cache(self):
        gemini_clear_cache()

    def get_cache_stats(self) -> Dict[str, int]:
        return gemini_get_cache_stats()

    def set_cache_enabled(self, enabled: bool):
        self.enable_cache = enabled

    def file_upload(self, file_path: str) -> Part:
        file = self.client.files.upload(file=file_path)
        # with tqdm(total=1000, desc="Processing video", unit="percent") as pbar:
        for i in range(1000):
            try:
                file_status = self.client.files.get(name=file.name)
                if file_status.state == "ACTIVE":
                    # pbar.update(1000 - pbar.n)  # Complete the progress bar
                    break
                elif file_status.state == "FAILED":
                    pass
            except Exception as e:
                # pbar.update(1)
                if file is None:
                    time.sleep(10)
                else:
                    break
        time.sleep(0.5)
        assert file is not None, "File is None"
        return file

    def count_tokens(self, video_path: str) -> int:
        if self.use_vertexai:
            return 0
        else:
            file = self._retry_with_exponential_backoff(self.file_upload, video_path)
            try: 
                count = self.client.models.count_tokens(
                model="gemini-2.5-flash",
                    contents=[file],
                ).total_tokens
                print(f"Number of tokens: {count}")
                try:
                    self.client.files.delete(name=file.name)
                except Exception as e:
                    pass
                return count
            except Exception as e:
                return 0
    
    def process_media(self, media_path: str) -> str:
        if self.use_vertexai:
            # upload the video to gcp bucket
            media_name = os.path.basename(media_path)
            url = upload_to_gcp_bucket(media_path, WEB_SEARCH.get("gcp_bucket"), media_name)
            return url
        else:
            media = self._retry_with_exponential_backoff(self.file_upload, media_path)
            return media

    def _retry_with_exponential_backoff(self, func, *args, **kwargs):
        for attempt in range(8):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"Error in gemini api: {str(e)}")
                print(f"Error type: {type(e).__name__}")
                time.sleep(2**attempt)

    def get_response(
        self,
        query: str,
        media_paths: List[str] = None,
        media_type: str = None,
        model_name: str = "gemini-2.5-flash",
        temperature=None,
    ) -> str:
        # Normalize model for caching and API usage
        medias = []
        if ":" in model_name:
            model_name = model_name.split(":")[1]
        # Attempt cache hit before any uploads
        if self.enable_cache:
            cached = gemini_get_cached_response(
                query=query,
                model=model_name,
                temperature=TEMPERATURE if temperature is None else float(temperature),
                media_paths=media_paths if media_paths is not None else [],
                media_type=media_type,
                use_vertexai=self.use_vertexai,
            )
            if cached is not None:
                return cached
        # check if the video_path is a youtube URL
        if media_paths is not None and len(media_paths) > 0:
            with ThreadPoolExecutor(max_workers=min(len(media_paths), 8)) as executor:
                futures = []
                for media_path in media_paths:
                    assert os.path.exists(media_path), "Vfile does not exist: {}".format(media_path)
                    futures.append(executor.submit(self.process_media, media_path))

                for media_path, future in zip(media_paths, futures):
                    try:
                        media_url = future.result()
                        if self.use_vertexai:
                            medias.append(
                                Part.from_uri(
                                    file_uri=media_url,
                                    mime_type=f"{media_type}/{media_path.split('.')[-1]}",
                                )
                            )
                        else:
                            medias.append(media_url)
                    except Exception as e:
                        if "409" in str(e) and "metadata" in str(e):
                            # Retry once on metadata conflict
                            time.sleep(1)
                            media_url = self.process_media(media_path)
                            if self.use_vertexai:
                                medias.append(
                                    Part.from_uri(
                                        file_uri=media_url,
                                        mime_type=f"{media_type}/{media_path.split('.')[-1]}",
                                    )
                                )
                            else:
                                medias.append(media_url)
                        else:
                            raise e

            if len(medias) > 1:
                content = medias
                if media_type == "video":
                    content = [medias[0]]
                content.extend([query])
            elif len(medias) == 1:
                content = [medias[0], query]
            else:
                content = [query]
        else:
            content = [query]
        
        if temperature is None:
            temperature = TEMPERATURE

        # exit()
        for attempt in range(8):
            try:
                # print(f"""Number of tokens: {self.client.models.count_tokens(
                #     model=model_name, contents=content
            # )}""")
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=content,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(include_thoughts=False),
                        temperature=temperature,
                    ),
                )
                break
            except Exception as e:
                if "400 FAILED_PRECONDITION" in str(e) and "not in an ACTIVE" in str(e) and attempt < 7:
                    time.sleep(10)
                elif "exceeds the maximum number of tokens allowed" in str(e):
                    return "No response from the model: {}".format(e)
                elif "403 PERMISSION_DENIED" in str(e) and attempt < 7:
                    time.sleep(10)
                else:
                    print(f"Error: {str(e)}")
                    print(f"Error type: {type(e).__name__}")
                    if attempt >= 7:  # Last attempt
                        return "No response from the model: {}".format(str(e))
                    # Exponential backoff: 1s, 2s, 4s, 8s, 16s, 32s, 64s, 128s, 256s
                    delay = min(2**attempt, 256)  # Cap at 256 seconds
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)  # Wait before retrying

        if (
            not self.use_vertexai
            and media_paths is not None
            and len(medias) > 0
        ):
            for media in medias:
                try:
                    self.client.files.delete(name=media.name)
                except Exception as e:
                    pass

        result = {"thoughts": "", "answer": ""}
        try:
            for part in response.candidates[0].content.parts:
                if not part.text:
                    continue
                if part.thought:
                    result["thoughts"] += part.text + "\n\n"
                else:
                    result["answer"] = part.text
        except Exception as e:
            result["answer"] = response.text
            result["thoughts"] = ""

        # Store in cache after successful response
        if self.enable_cache:
            try:
                gemini_set_cached_response(
                    query=query,
                    model=model_name,
                    temperature=temperature,
                    response=result,
                    media_paths=media_paths if media_paths is not None else [],
                    media_type=media_type,
                    use_vertexai=self.use_vertexai,
                )
            except Exception:
                pass

        return result
