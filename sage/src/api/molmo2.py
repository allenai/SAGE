from typing import List, Optional, Union, Any
import torch
from PIL import Image
import numpy.typing as npt
import logging
import os
from sage.src.models.molmo2.vision_process import process_vision_info

logger = logging.getLogger(__name__)

from sage.src.models.molmo2.processing import VideoMolmo2Processor, Molmo2VideoProcessor
from sage.src.models.molmo2 import Molmo2ForConditionalGeneration as VideoMolmo2ForConditionalGeneration
from transformers import AutoTokenizer
from sage.src.api.base_api import BaseAPI

from vllm import SamplingParams, ModelRegistry
from vllm.model_executor.models.registry import _MULTIMODAL_MODELS
from vllm.multimodal.video import VideoLoader, VIDEO_LOADER_REGISTRY

try:
    from sage.src.models.molmo2.vllm.modeling import VideoMolmo2ForConditionalGeneration as Molmo2ForConditionalGeneration
    ModelRegistry.register_model("Molmo2ForConditionalGeneration", Molmo2ForConditionalGeneration)
    _MULTIMODAL_MODELS["Molmo2ForConditionalGeneration"] = ("molmo2", "Molmo2ForConditionalGeneration")
    os.environ["VLLM_USE_V1"] = "0"
    os.environ["VLLM_VIDEO_LOADER_BACKEND"] = "molmo2"
except:
    print("Import error: Molmo2ForConditionalGeneration not found")

@VIDEO_LOADER_REGISTRY.register("molmo2")
class Molmo2VideoBackend(VideoLoader):

    @classmethod
    def get_candidate_sampling_fps(
        cls,
        video_fps: float,
        sampling_fps: float,
        max_fps: float = 8.0,
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

    @classmethod
    def sample_times(
        cls,
        duration: float,
        max_frames: int,
        frame_sample_mode: str,
        max_fps: int | None,
        candidate_sampling_fps: Optional[List[float]] = None,
        **kwargs,
    ) -> npt.NDArray:

        if frame_sample_mode == "fps":
            assert candidate_sampling_fps is not None
            # Try larger and larger FPSs until we hit one that can't span the video
            sampling_fps = candidate_sampling_fps[0]
            for candidate_fps in candidate_sampling_fps[1:]:
                if max_frames / candidate_fps < duration:
                    break
                sampling_fps = candidate_fps
            times = np.arange(0, max_frames) / sampling_fps
            times = times[times < duration]
            return times
        elif frame_sample_mode == "uniform_last_frame":
            if max_fps is not None:
                max_duration = (max_frames-1) / max_fps  # -1 to include the last frame
                if max_duration < duration:
                    times = np.linspace(
                        0, duration, num=max_frames, endpoint=True, dtype=np.float64
                    )
                else:
                    times = np.arange(0.0, stop=duration, step=1/max_fps)
                    times = np.concatenate([times, [duration]], axis=0)
                    assert len(times) <= max_frames
            else:
                times = np.linspace(
                    0, duration, num=max_frames, endpoint=True, dtype=np.float64
                )
            return times
        else:
            raise NotImplementedError(frame_sample_mode)

    @classmethod
    def load_bytes(
        cls,
        data: bytes,
        max_frames: int = 128,
        frame_sample_mode: str = "uniform_last_frame",
        sampling_fps: int = 2,
        max_fps: int = 2,
        **kwargs,
    ) -> dict[str, Any]:
        import torchcodec

        decoder = torchcodec.decoders.VideoDecoder(data, num_ffmpeg_threads=1)
        video_fps = decoder.metadata.average_fps

        candidate_sampling_fps: Optional[List[float]] = None
        if frame_sample_mode == "fps":
            candidate_sampling_fps = cls.get_candidate_sampling_fps(video_fps, sampling_fps)
        
        # If the first frame starts at > 0, we effectively clip the video starting at that time
        # since (most) video players would also skip to that time
        time_offset = decoder.metadata.begin_stream_seconds_from_content
        # Note this duration does assume we started playing at `time_offset`
        duration = decoder.metadata.duration_seconds

        target_timestamps = cls.sample_times(
            duration,
            max_frames,
            frame_sample_mode,
            max_fps,
            candidate_sampling_fps,
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

        out = {
            "frames": frames.data.numpy().transpose(0, 2, 3, 1),
            "timestamps": target_timestamps,
        }
        return out


class Molmo2(BaseAPI):
    def __init__(
        self,
        model_name: str = None,
        vllm_engine: object = None,
        processor: VideoMolmo2Processor = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 16384,
        temperature: float = 0.7,
        top_p: float = 0.9,
        is_rl_train_mode: bool = False,
        use_video: bool = True,
        tool_call_clients: List[object] = None,
    ):
        """
        Initialize the Molmo2 model.

        Args:
            model_name: Name of the model on Hugging Face Hub
            device: Device to run the model on ('cuda' or 'cpu')
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
        """
        if processor is None and ":" in model_name:
            logger.info(f"Loading Molmo2 model from {model_name}")
            model_path = model_name.split(":")[-1]
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            video_processor = Molmo2VideoProcessor()
            processor = VideoMolmo2Processor(
                video_processor=video_processor,
                tokenizer=tokenizer
            )
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
            model_class=VideoMolmo2ForConditionalGeneration,
        )
    
    def _prepare_input(
        self, 
        prompt: str, 
        media: Optional[Union[str, Image.Image, List[Image.Image]]] = [], 
        media_type: Optional[str] = None,
        system_prompt: Optional[str] = None,\
        eval_style: Optional[str] = None,
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
        
        if media_type == "video":
            media = [media[0]]
        
        if media_type is not None and self.use_video:
            if media_type == "video":
                content = [
                    {
                        "type": "video",
                        "video": media[0],
                        "backend": "decord",
                        "max_frames": 128,
                        "frame_sample_mode": "uniform_last_frame"
                    }
                ]
            else:
                content = [
                    {
                        "type":"image",
                        "image": media_path
                    }
                    for media_path in media
                ]
            
        else:
            if not self.use_video:
                print("Not using video")
            content = []
        
        if system_prompt is not None:
            text_msg = {"type": "text", "text": system_prompt + "\n" + prompt}
        else:
            text_msg = {"type": "text", "text": prompt}
            
        if eval_style is not None:
            text_msg["style"] = eval_style
        content.append(text_msg)
        
        messages = [
            {
            "role": "user",
            "content": content
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        inputs = inputs.to(self.device)

        if self.vllm_engine is not None:
            if media_type is not None and self.use_video:
                multimodal_inputs = [
                    {"prompt": text, "multi_modal_data": {media_type: video_inputs if media_type == "video" else image_inputs}}
                ]
            else:
                if not self.use_video:
                    print("Not using video")
                multimodal_inputs = [text]
        else:
            multimodal_inputs = None

        return inputs, multimodal_inputs
    
    def get_response(
        self,
        prompt: str,
        media: Optional[Union[str, Image.Image, List[Image.Image]]] = None,
        media_type: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        return_ids: bool = False,
        temperature: float = None,
        eval_style: Optional[str] = None,
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
        if media is not None and len(media) > 0:
            for media_path in media:
                if not os.path.exists(media_path):
                    raise FileNotFoundError(f"Media file does not exist: {media_path}, in the passed media list: {media}")
        if max_new_tokens is None:
            max_new_tokens = self.max_length

        inputs, multimodal_inputs = self._prepare_input(prompt, media, media_type, system_prompt, eval_style)

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

            generated_ids = self.vllm_engine.generate(multimodal_inputs, sampling_params=sampling_params, use_tqdm=False, **kwargs)[0]
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
        
        if return_ids:
            return output_text, generated_ids_trimmed, inputs.input_ids[0], torch.cat([inputs.attention_mask[0], torch.ones(len(generated_ids_trimmed)).to(self.device)], dim=0)
        else:
            return output_text

if __name__ == "__main__":
    molmo2 = Molmo2(
        model_name="/root/home/checkpoints/sage/molmo2_3b",
    )
    response = molmo2.get_response(
        prompt="What is the capital of France?",
        media=["https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png"],
        media_type="image",
    )
    print(response)