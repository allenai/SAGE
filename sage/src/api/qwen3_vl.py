from typing import List, Optional, Union, Dict, Any
import torch
from PIL import Image
import logging
import os
import gc

from sage.src.api.base_api import BaseAPI
logger = logging.getLogger(__name__)

from vllm import LLM, SamplingParams
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from sage.src.models.qwen_vl.vision_process import process_vision_info


TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.0))
print(f"TEMPERATURE: {TEMPERATURE}")

USE_VLLM = True
MAX_GEN_TOKENS = int(os.environ.get("MAX_GEN_TOKENS", "32768"))
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "128"))
MIN_FRAMES = int(os.environ.get("MIN_FRAMES", "128"))
MAX_TOKENS_PER_FRAME = int(os.environ.get("MAX_TOKENS_PER_FRAME", "192"))
MIN_TOKENS_PER_FRAME = int(os.environ.get("MIN_TOKENS_PER_FRAME", "128"))
FPS = float(os.environ.get("FPS", "2.0"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1"))

def prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Tokenize to check token count
    tokenized = processor.tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    num_tokens = tokenized.input_ids.shape[-1]

    max_context_tokens = MAX_GEN_TOKENS
    keep_first = 1024  # always keep first 1024 tokens

    if num_tokens > max_context_tokens:
        logger.warning(
            f"⚠️ Input sequence too long ({num_tokens} > {max_context_tokens}). "
            f"Truncating while keeping first {keep_first} tokens."
        )
        truncated_ids = torch.cat([
            tokenized.input_ids[:, :keep_first],
            tokenized.input_ids[:, - (max_context_tokens - keep_first):]
        ], dim=-1)
        text = processor.tokenizer.decode(
            truncated_ids[0],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    # Process multimodal (image/video) inputs
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )

    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs

    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs
    }


class Qwen3_VL(BaseAPI):
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
    ):
        """
        Initialize the Qwen3-VL model.

        Args:
            model_name: Name of the model on Hugging Face Hub
            device: Device to run the model on ('cuda' or 'cpu')
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
        """
        if processor is None and ":" in model_name:
            model_path = model_name.split(":")[-1]
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        super().__init__(
            model_name=model_name,
            vllm_engine=vllm_engine,
            processor=processor,
            device=device,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            is_rl_train_mode=is_rl_train_mode,
            use_video=use_video,
            tool_call_clients=tool_call_clients,
            model_class=Qwen3VLForConditionalGeneration,
        )


    def _prepare_input(
        self, 
        prompt: str, 
        media: Optional[Union[str, Image.Image, List[Image.Image]]] = [], 
        media_type: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> Any:
        """
        Prepare media input (image or video) for the model.

        Args:
            media: Media input (path to image/video, PIL Image, or list of PIL Images)
            prompt: Text prompt

        Returns:
            Formatted query for the model
        """
        if media_type is not None:
            assert media_type in ["image", "video"], f"Invalid media type: {media_type}"

        if system_prompt is not None:
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                }
            ]
        else:
           messages = [
                {
                    "role": "system",
                    "content": "Respond in a concise and direct manner to the user's question."
                }
            ]
        
        if media_type == "video":
            media = [media[0]]
        
        if media_type is not None and self.use_video:
            content = [
                {
                    "type": media_type,
                    media_type: media_path,
                    "max_frames": MAX_FRAMES,
                    "min_frames": MIN_FRAMES,
                    "max_pixels": MAX_TOKENS_PER_FRAME * 28 * 28,
                    "min_pixels": MIN_TOKENS_PER_FRAME * 28 * 28,
                    "fps": FPS,
                }
                for media_path in media
            ]
        else:
            if not self.use_video:
                print("Not using video")
            content = []
        
        content.append({"type": "text", "text": prompt})
        
        messages.append({
            "role": "user",
            "content": content
        })
        
        inputs = None
        if self.vllm_engine is None:
            # Preparation for inference
            text = self.processor.apply_chat_template(
                messages, tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True, use_cache=True)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            )
            inputs = inputs.to(self.device)

        return inputs, messages
    
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
        """
        Generate text based on the prompt and optional media input.

        Args:
            prompt: Text prompt for the model
            media: Optional media input (can be path to image/video, PIL Image, or list of PIL Images)
            max_new_tokens: Maximum number of new tokens to generate

        Returns:
            Generated text response
        """
        kwargs.pop("eval_style", None)
        if media is not None and len(media) > 0:
            for media_path in media:
                if not os.path.exists(media_path):
                    raise FileNotFoundError(f"Media file does not exist: {media_path}, in the passed media list: {media}")
        if max_new_tokens is None:
            max_new_tokens = self.max_length
        

        inputs, messages = self._prepare_input(prompt, media, media_type, system_prompt)

        if self.vllm_engine is not None:
            if "generation_kwargs" in kwargs:
                generation_kwargs = kwargs.pop("generation_kwargs")
            else:
                generation_kwargs = self.generation_kwargs

            if temperature is not None:
                generation_kwargs["temperature"] = temperature
            else:
                temperature = self.temperature

            if "sampling_params" in kwargs:
                sampling_params = kwargs.pop("sampling_params")
            else:
                sampling_params = SamplingParams(**generation_kwargs)
            
            inputs = [prepare_inputs_for_vllm(message, self.processor) for message in [messages]]
            
            generated_ids = self.vllm_engine.generate(inputs, sampling_params=sampling_params, use_tqdm=False, **kwargs)[0]
            generated_ids_trimmed = generated_ids.outputs[0].token_ids
            output_text = generated_ids.outputs[0].text
        else:
            if temperature is None:
                temperature = self.temperature
            
            if temperature == 0.0:
                kwargs["do_sample"] = False
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, **kwargs)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ][0]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            output_text = "".join(output_text)

        return output_text

if __name__ == "__main__":
    qwen_vl = Qwen3_VL(
        model_name="/root/home/checkpoints/sage/Qwen3-VL-8B-Instruct",
    )
    response = qwen_vl.get_response(
        prompt="What is happening in the video?",
        media=["/root/home/sHz6X0fLr9U.mp4"],
        media_type="video",
    )
    print(response)