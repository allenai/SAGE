# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Union, Optional
import re
from omegaconf import DictConfig

from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import RLHFDataset
import os

IS_MOLMO2 = os.environ.get("IS_MOLMO2", "False")
IS_MOLMO2 = IS_MOLMO2.lower() == "true"
VIDEO_DIR = os.environ.get("VIDEO_DIR", "data/videos")

class SAGE_RLDataset(RLHFDataset):
    """
    Dataset for tool use in RLHF
    """
    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        super().__init__(data_files, tokenizer, config, processor)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = self.dataframe[item]
        messages = self._build_messages(row_dict)
        model_inputs = {}

        if IS_MOLMO2:
            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        if self.processor is not None:
            from rl_train.utils.dataset.vision_utils import process_image, process_video, process_molmo2

            multi_modal_data = {}
            raw_inputs = {}

            images = None
            if self.image_key in row_dict:
                images = [process_image(image) for image in row_dict.pop(self.image_key)]
                multi_modal_data["image"] = images

            in_videos = None
            video_paths = None
            if self.video_key in row_dict:
                video_paths = [os.path.join(VIDEO_DIR, row_dict.get(self.video_key) + ".mp4")]
                if IS_MOLMO2:
                    def _create_video_dict(video_path):
                        return {
                            "backend": "decord",
                            "frame_sample_mode": "uniform_last_frame",
                            "max_frames": 128,
                            "type": "video",
                            "video": video_path
                        }
                    videos = [process_molmo2(_create_video_dict(video_path)) for video_path in video_paths]
                    in_videos = videos
                else:
                    def _create_video_dict(video_path):
                        return {
                            "type": "video",
                            "video": video_path
                        }
                    videos = [process_video(_create_video_dict(video_path)) for video_path in video_paths]
                    in_videos = [v[0].numpy() if isinstance(v, tuple) else v.numpy() for v in videos]
                
                multi_modal_data["video"] = videos

                raw_inputs["question"] = [row_dict.pop("question")]
                raw_inputs["gt_answer"] = [row_dict.pop("gt_answer")]
                raw_inputs["video_duration"] = [row_dict.pop("duration_seconds")]
                raw_inputs["video_duration_timestamp"] = [row_dict.pop("video_duration")]
            
            model_inputs = self.processor(text=[raw_prompt], images=images, videos=in_videos, return_tensors="pt")

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            raw_inputs["video_path"] = video_paths
            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data
            row_dict["raw_inputs"] = raw_inputs
            row_dict["multi_modal_inputs"] = dict(model_inputs)

            # second_per_grid_ts isn't used for training, just for mrope
            row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if not IS_MOLMO2 and self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)

        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        row_dict["raw_prompt"] = raw_prompt

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict