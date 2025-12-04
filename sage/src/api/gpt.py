from openai import OpenAI
# from openai import AzureOpenAI
import os
import base64
from sage.utils.utils import API_KEYS
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from sage.src.api.gpt_cache import get_cached_response, set_cached_response

WEBUI_API_SUFFIX = "/v1"

TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.0))

class GPT:

    def __init__(self, enable_cache=True):
        self.client = OpenAI(
            api_key=API_KEYS.get("openai", "None"),
        )
        # Cache configuration
        self.enable_cache = enable_cache

    def get_models(self):
        return self.client.models.list()

    def get_response(self, prompt, image_urls=None, model="gpt-4o", history=[]):
        # Check cache first if enabled
        if ":" in model:
            model = model.split(":")[1]
        if self.enable_cache:
            cached_result = get_cached_response(prompt, model, TEMPERATURE, image_urls, history)
            if cached_result is not None:
                return cached_result

        new_message_content = []

        if image_urls is not None:
            assert isinstance(image_urls, list), "image_urls must be a list"

            def process_image(image_url):
                assert isinstance(image_url, str), "image_url must be a string"
                assert (
                    image_url.startswith("http://")
                    or image_url.startswith("https://")
                    or os.path.isfile(image_url)
                    or image_url.startswith("file://")
                ), "image_url must be a URL or a local file path"
                if os.path.isfile(image_url) or image_url.startswith("file://"):
                    local_path = image_url.replace("file://", "")
                    with open(local_path, "rb") as image_file:
                        time_stamp = image_file.name.split("_")[-1].split(".")[0]
                        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                        image_url = f"data:image/jpeg;base64,{encoded_string}"
                return [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": f"frame_timestamp: {time_stamp}"},
                ]

            with ThreadPoolExecutor(max_workers=min(len(image_urls), 256)) as executor:
                futures = []
                for image_url in image_urls:
                    futures.append(executor.submit(process_image, image_url))
                for image_url, future in zip(image_urls, futures):
                    processed_images = future.result()
                    new_message_content.extend(processed_images)

        new_message_content.append({"type": "text", "text": prompt})
        messages = history + [{"role": "user", "content": new_message_content}]
        return_messages = history + [{"role": "user", "content": [new_message_content[0]]}]

        try:
            if model == "gpt-5":
                kwargs = {}
            else:
                kwargs = {
                    "temperature": TEMPERATURE,
                }
            response = self.client.chat.completions.create(model=model, messages=messages, **kwargs)
            # Check if 'choices' exist and are valid
            if not hasattr(response, "choices") or len(response.choices) == 0:
                raise Exception("Request failed: No valid choices in response")

            if hasattr(response, "error"):
                raise Exception(f"Request failed with error: {response['error']}")
        except Exception as e:
            return str(e), history

        return_messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response.choices[0].message.content}],
            }
        )

        # Cache the response if caching is enabled
        if self.enable_cache:
            set_cached_response(prompt, model, TEMPERATURE, response.choices[0].message.content, return_messages, image_urls, history)

        return response.choices[0].message.content, return_messages

    def clear_cache(self):
        """Clear all cached GPT responses."""
        from sage.src.api.gpt_cache import clear_cache
        clear_cache()

    def get_cache_stats(self):
        """Get cache statistics."""
        from sage.src.api.gpt_cache import get_cache_stats
        return get_cache_stats()

    def set_cache_enabled(self, enabled: bool):
        """Enable or disable caching."""
        self.enable_cache = enabled
