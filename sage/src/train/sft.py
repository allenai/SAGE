# Copyright 2024. All rights reserved.
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

import torch
from datasets import load_dataset
from transformers import logging
import warnings
import os
import signal
from contextlib import contextmanager
# Disable transformers warnings but keep progress bar
os.environ["TRANSFORMERS_VERBOSITY"] = "warning"
logging.set_verbosity_warning()
warnings.filterwarnings("ignore", message=".*unused.*")
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
)
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTTrainer,
    SFTConfig,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
)
from sage.src.models.qwen_vl.vision_process import process_vision_info
from sage.src.models.molmo2.vision_process import process_vision_info as process_vision_info_molmo2
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration
from sage.src.models.molmo2 import Molmo2ForConditionalGeneration

from typing import List, Dict, Any
import os
import glob

@contextmanager
def timeout(duration):
    """Context manager for timing out operations after specified duration in seconds."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {duration} seconds")
    
    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    
    try:
        yield
    finally:
        # Restore the old signal handler
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)

def find_latest_checkpoint(output_dir):
    """
    Find the latest checkpoint in the output directory.
    
    Args:
        output_dir (str): Path to the output directory
        
    Returns:
        str or None: Path to the latest checkpoint, or None if no checkpoints found
    """
    if not os.path.exists(output_dir):
        return None
    
    # Look for checkpoint directories (usually named like "checkpoint-{step}")
    checkpoint_pattern = os.path.join(output_dir, "checkpoint-*")
    checkpoint_dirs = glob.glob(checkpoint_pattern)
    
    if not checkpoint_dirs:
        return None
    
    # Sort by step number to find the latest
    def extract_step(checkpoint_path):
        try:
            return int(checkpoint_path.split("-")[-1])
        except (ValueError, IndexError):
            return 0
    
    latest_checkpoint = max(checkpoint_dirs, key=extract_step)
    print(f"Found latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint


def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate batch of examples for training."""
    texts = []
    image_inputs = []
    video_inputs = []

    for i, example in enumerate(examples):
        try:
            messages = []
            for msg in example["messages"]:
                content = msg["content"]
                contents = []
                for ele in content:
                    if ele["type"] == "text":
                        ele.pop("video", None)
                        ele.pop("image", None)
                    elif ele["type"] == "video":
                        ele.pop("text", None)
                        ele.pop("image", None)
                    elif ele["type"] == "image":
                        ele.pop("video", None)
                        ele.pop("text", None)
                    contents.append(ele)
                msg["content"] = contents
                messages.append(msg)
            texts.append(processor.apply_chat_template(messages, tokenize=False))
            image_input, video_input, video_kwargs = process_vision_info(
                messages, return_video_kwargs=True, use_cache=False
            )
            if image_input is not None:
                image_inputs.append(image_input)
            if video_input is not None:
                video_inputs.append(video_input[0])
        except Exception as e:
            raise ValueError(f"Failed to process example {i}: {e}")

    if len(video_inputs) == 0:
        video_inputs = None
    if len(image_inputs) == 0:
        image_inputs = None
        
    with timeout(60):  # 5 minutes timeout
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
            **video_kwargs
        )
    labels = inputs["input_ids"].clone()

    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Handle visual tokens based on processor type
    visual_tokens = [151652, 151653, 151654, 151655, 151656]
    
    count = 0
    for visual_token_id in visual_tokens:
        count += (labels == visual_token_id).sum()
        labels[labels == visual_token_id] = -100

    inputs["labels"] = labels

    if inputs["input_ids"].shape[1] > 16384:
        texts = []
        print("Using fallback msg...")
        msg = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful assistant."
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Hello, how can I help you today?"
                    }
                ]
            }
        ]
        texts.append(processor.apply_chat_template(msg, tokenize=False))
        inputs = processor(
            text=texts,
            return_tensors="pt",
            padding=True,
        )
        labels = torch.ones_like(inputs["input_ids"]) * -100
        inputs["labels"] = labels
    return inputs

def molmo2_collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate batch of examples for training."""
    texts = []
    video_inputs = []
    num_tokens_list = []  # Store num_tokens for each example

    for i, example in enumerate(examples):
        try:
            messages = []
            system_text = example["messages"][0]["content"][0]["text"]
            num_tokens = 0  # Initialize for this example
            for msg in example["messages"][1:]:
                content = msg["content"]
                contents = []
                for ele in content:
                    if ele["type"] == "text":
                        ele.pop("video", None)
                        if msg["role"] == "user":
                            ele["text"] = system_text + "\n\n" + ele["text"]
                        elif msg["role"] == "assistant":
                            ele["text"] = ele["text"] + processor.tokenizer.decode(processor.tokenizer.eos_token_id)
                            num_tokens = len(processor.tokenizer(ele["text"])["input_ids"])
                    elif ele["type"] == "video":
                        ele.pop("text", None)
                        ele["backend"] = "decord"
                        ele["max_frames"] = 128
                        ele["frame_sample_mode"] = "uniform_last_frame"
                    contents.append(ele)
                msg["content"] = contents
                messages.append(msg)
            texts.append(processor.tokenizer.apply_chat_template(messages, tokenize=False))
            num_tokens_list.append(num_tokens)  # Store num_tokens for this example
            video_input = process_vision_info_molmo2(messages, use_cache=False)[1]
            if video_input is not None:
                video_inputs.append(video_input[0])
        except Exception as e:
            raise ValueError(f"Failed to process example {i}: {e}")

    if len(video_inputs) == 0:
        video_inputs = None
    
    inputs = processor(
        text=texts,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    )
    labels = inputs["input_ids"].clone()
    
    # Mask the last num_tokens for each sequence in the batch
    for i, num_tokens in enumerate(num_tokens_list):
        if num_tokens > 0:
            labels[i, :-num_tokens] = -100
    
    inputs["labels"] = labels

    if inputs["input_ids"].shape[1] > 16384:
        texts = []
        print("Using fallback msg...")
        msg = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful assistant."
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Hello, how can I help you today?"
                    }
                ]
            }
        ]
        texts.append(processor.apply_chat_template(msg, tokenize=False))
        inputs = processor(
            text=texts,
            return_tensors="pt",
            padding=True,
        )
        labels = torch.ones_like(inputs["input_ids"]) * -100
        inputs["labels"] = labels
    return inputs


if __name__ == "__main__":
    # Parse arguments
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()

    # Configure training args
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    # Setup model
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )

    # Model initialization
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map(),
    )

    if "Qwen2.5-VL" in model_config.model_name_or_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_config.model_name_or_path, **model_kwargs
        )
    elif "Qwen3-VL" in model_config.model_name_or_path:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_config.model_name_or_path, **model_kwargs
        )
    elif "molmo2" in model_config.model_name_or_path:
        model_kwargs.pop("trust_remote_code")
        model = Molmo2ForConditionalGeneration.from_pretrained(
            model_config.model_name_or_path, **model_kwargs
        )
    else:
        model = AutoModelForVision2Seq.from_pretrained(model_config.model_name_or_path, **model_kwargs)

    if "molmo2" in model_config.model_name_or_path:
        from sage.src.models.molmo2 import VideoMolmo2Processor
        from sage.src.models.molmo2.video_processing import Molmo2VideoProcessor
        from transformers import AutoTokenizer
        
        # Load tokenizer and video processor separately
        tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
        video_processor = Molmo2VideoProcessor()
        
        # Create VideoMolmo2Processor instance manually
        processor = VideoMolmo2Processor(
            video_processor=video_processor,
            tokenizer=tokenizer
        )
    else:
        processor = AutoProcessor.from_pretrained(
            model_config.model_name_or_path,
            trust_remote_code=model_config.trust_remote_code,
        )

    # Prepare dataset
    prepared_dataset_train = load_dataset(script_args.dataset_name, split="train")
    prepared_dataset_eval = load_dataset(script_args.dataset_name, split="val")

    for name, param in model.named_parameters():
        if "visual" in name or "vision_backbone" in name:
            param.requires_grad = False
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=prepared_dataset_train,
        eval_dataset=prepared_dataset_eval,
        data_collator=collate_fn if "molmo2" not in model_config.model_name_or_path else molmo2_collate_fn,
        peft_config=get_peft_config(model_config),
        processing_class=processor,
    )

    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
    from deepspeed.runtime.zero.config import ZeroStageEnum
    from deepspeed.utils.tensor_fragment import fragment_address
    from deepspeed.runtime.fp16.loss_scaler import LossScaler
    from numpy.core.multiarray import _reconstruct
    import numpy as np
    from numpy.dtypes import UInt32DType

    torch.serialization.add_safe_globals([
        DeepSpeedZeroOptimizer,
        ZeroStageEnum,
        fragment_address,
        LossScaler,
        _reconstruct,
        np.ndarray,
        np.dtype,
        np.generic,
        np.bool_, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
        np.float16, np.float32, np.float64,
        UInt32DType
        # *[getattr(np, t) for t in dir(np) if isinstance(getattr(np, t), type)]
    ])

    # Check for existing checkpoints in output directory
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        print(f"Resuming from specified checkpoint: {checkpoint}")
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        # Check if there are existing checkpoints in the output directory
        existing_checkpoint = find_latest_checkpoint(training_args.output_dir)
        if existing_checkpoint is not None:
            print(f"Found existing checkpoint in output directory: {existing_checkpoint}")
            print("Resuming training from the latest checkpoint...")
            trainer.train(resume_from_checkpoint=existing_checkpoint)
        else:
            print("No existing checkpoints found. Starting training from scratch...")
            trainer.train()

    # Save final model
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    if trainer.accelerator.is_main_process:
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    # Cleanup
    del model
    del trainer
    torch.cuda.empty_cache()
