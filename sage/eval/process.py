import json
import os
# Suppress FFmpeg/av1 codec warnings - set before any video processing imports
os.environ.setdefault("AV_LOG_FORCE_NOCOLOR", "1")
os.environ.setdefault("AV_LOG_LEVEL", "error")  # Only show errors, not warnings (can use "quiet" to suppress more)
import argparse
from tqdm import tqdm
import time
from typing import Dict, List, Tuple, Optional, Any
import subprocess
import wandb
import pandas as pd
from datasets import DatasetDict, load_dataset
import signal
from contextlib import contextmanager
from transformers import logging
import warnings
# Disable transformers warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.set_verbosity_error()
warnings.filterwarnings("ignore", message=".*unused.*")
# Suppress FFmpeg/av1 codec warnings in Python warning system
warnings.filterwarnings("ignore", message=".*av1.*")
warnings.filterwarnings("ignore", message=".*Missing Sequence Header.*")
warnings.filterwarnings("ignore", message=".*mmco: unref short failure.*")
warnings.filterwarnings("ignore", message=".*h264.*")
warnings.filterwarnings("ignore", message=".*codec.*")
# Import both models
from sage.src.context_vlm import ContextVLM
from sage.main import SAGE
from sage.utils.utils import (
    SAMPLED_BASELINE_MINERVA_PROMPT,
    SAMPLED_BASELINE_PROMPT,
)
import math
from sage.src.functions.utils.transcribe import transcribe_video
import hashlib
import requests
from time import sleep

from sage.src.functions.utils.utils import check_api_health
from sage.src.functions.utils.temporal import get_video_duration

TOOL_CALL_MODEL = os.environ.get("TOOL_CALL_MODEL", "None")
VLLM_CLIENT_URL = os.environ.get("VLLM_CLIENT_URL", "None")
TRANSCRIBE_API_URL = os.environ.get("TRANSCRIBE_API_URL", "None")

def check_tool_call_model():
    if TOOL_CALL_MODEL is not None and TOOL_CALL_MODEL != "None":
        # print("Checking tool call model status...")
        urls = [url.strip() for url in VLLM_CLIENT_URL.split(",") if url.strip()]
        for url in urls:
            check_api_health(url, "VLLM client")

USE_ASR = os.environ.get("USE_ASR", "True")
USE_ASR = USE_ASR.lower() == "true"

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    if len(lst) == 0:
        return [[] for _ in range(n)]
    if len(lst) < n:
        # If list is shorter than n, create chunks with at most 1 item each
        chunks = []
        for i in range(n):
            if i < len(lst):
                chunks.append([lst[i]])
            else:
                chunks.append([])
        return chunks
    
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    if k >= len(chunks):
        return []
    return chunks[k]


@contextmanager
def timeout(seconds):
    """Context manager for timeout functionality."""
    def signal_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    # Set the signal handler and a 5-second alarm
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class Processor:
    def __init__(
        self,
        model_name: str = "sage:allenai/SAGE-MM-Qwen3-VL-8B-SFT_RL",
        num_sampled_frames: int = 128,
        gpu_idx: int = 0,
        benchmark: str = "sage_bench",
        tool_to_drop: str = None,
        timeout_seconds: int = 300,  # 5 minutes default timeout
        max_num_iterative_reasoner_calls: int = 10,
        use_gemini_as_tool: bool = False,
        use_video: bool = True,
    ):
        if not use_video:
            print("Not using video")
            
        if benchmark == "minerva_bench":
            self.benchmark = "minerva_bench"
            self.dataset_path = "data/minerva_videos/minerva.json"
        elif benchmark == "sage_bench":
            self.benchmark = "sage_bench"
            self.dataset_path = "allenai/SAGE-Bench"
        else:
            raise ValueError(f"Benchmark {benchmark} not supported")
        
        self.num_sampled_frames = num_sampled_frames
        self.model_name = model_name.lower()
        self.gpu_idx = gpu_idx
        self.entry = 0

        if "_" in tool_to_drop:
            drop_tool_call_files = tool_to_drop.split("_")
        else:
            drop_tool_call_files = [tool_to_drop] if tool_to_drop is not None else []

        if "sage" in model_name.lower() and ("gemini" in model_name.lower() or "gpt" in model_name.lower()):
                self.model = SAGE(
                    model_name, 
                    drop_tool_call_files=drop_tool_call_files, 
                    max_num_iterative_reasoner_calls=max_num_iterative_reasoner_calls,
                    use_video=use_video,
                )
                self.model.context_vlm.delete_client_files()
        elif "qwen" in model_name.lower() or "molmo2" in model_name.lower():
            self.model = SAGE(
                model_name, 
                drop_tool_call_files=drop_tool_call_files, 
                max_num_iterative_reasoner_calls=max_num_iterative_reasoner_calls,
                use_gemini_as_tool=use_gemini_as_tool,
                use_video=use_video,
            )
        elif "gemini" in model_name.lower():
            self.model = ContextVLM(api_type=model_name, use_video=use_video)
            self.model.delete_client_files()
        elif "gpt" in model_name.lower():
            self.model = ContextVLM(api_type=model_name, use_video=use_video)
        else:
            raise ValueError(f"Model name {model_name} not supported")
            
        self.output_file = f"{model_name.split(':')[0]}_{model_name.split('/')[-1]}_{self.benchmark}_results.jsonl"

        self.tools_so_far = {}
        self.timeout_seconds = timeout_seconds

        self.is_base_molmo2 = "molmo2" in model_name.lower() and "sage" not in model_name.lower()

    def load_videos(self) -> List[Dict]:
        if self.benchmark == "minerva_bench":
            return self.load_minerva_videos()
        elif self.benchmark == "sage_bench":
            return self.load_sage_bench_videos()
        else:
            raise ValueError(f"Benchmark {self.benchmark} not supported")
    

    def load_sage_bench_videos(self) -> List[Dict]:
        """Load and process video data from the JSON file."""
        
        videos = load_dataset(self.dataset_path, split="test")

        video_sets = []
        for idx, v in tqdm(enumerate(videos), total=len(videos), desc="Loading videos"):

            video_path = "data/sage_bench_videos/" + v["video_id"] + ".mp4"

            video_duration = get_video_duration(video_path)
            transcript_path = video_path.replace(".mp4", ".txt")
            if not os.path.exists(transcript_path) and ("sage" not in self.model_name):
                print(f"Transcript not found for {video_path}, transcribing...")
                try:
                    transcript = str(transcribe_video(video_path))
                     # save the transcript to a txt file
                    with open(transcript_path, "w") as f:
                        f.write(transcript)
                except Exception as e:
                    print(f"Error transcribing {video_path}: {e}")
                    transcript = ""
            elif "sage" not in self.model_name:
                transcript = open(transcript_path).read()
            else:
                transcript = ""

            question = v["question"]

            is_mcq = v["ques_type"] == "mcq"

            if "sage" not in self.model_name:
                question = (
                    SAMPLED_BASELINE_PROMPT.replace("<<<question>>>", question)
                )
                if USE_ASR:
                    question = question.replace("<<<asr>>>", transcript)
            
            video_sets.append(
                {
                    "id": hashlib.md5(f"{v['question']}|{video_path}".encode()).hexdigest(),
                    "path": video_path,
                    "question": question if not (self.is_base_molmo2 and is_mcq) else (question, "video_eval_multiple_choice"),
                    "answer": None,
                    "full_answer": v["gt_answer"],
                    "difficulty": v["difficulty"],
                    "modality": v["modality"],
                    "ques_type": v["ques_type"].replace("-", "_"),
                    "video_duration": video_duration,
                }
            )

        # Skip already processed videos
        if os.path.exists(self.output_file):
            with open(self.output_file, "r") as f:
                existing_ids = [json.loads(line)["id"] for line in f]
            video_sets = [v for v in video_sets if v["id"] not in existing_ids]

        return video_sets
    
    def load_minerva_videos(self) -> List[Dict]:
        """Load and process video data from the JSON file."""
        with open(self.dataset_path) as f:
            videos = json.load(f)

        ANSWER_ID_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}

        video_sets = []
        for idx, v in tqdm(enumerate(videos), total=len(videos), desc="Loading videos"):

            video_dir = os.path.join("data", "minerva_videos", v["video_id"])
            video_path = os.path.join(video_dir, v["video_id"] + ".mp4")

            transcript_path = video_path.replace(".mp4", ".txt")
            video_duration = get_video_duration(video_path)
            if "sage" not in self.model_name:
                question = v["question"]
                choices = (
                    f"(A) {v['answer_choice_0']}\n"
                    + f"(B) {v['answer_choice_1']}\n"
                    + f"(C) {v['answer_choice_2']}\n"
                    + f"(D) {v['answer_choice_3']}\n"
                    + f"(E) {v['answer_choice_4']}"
                )
                transcript = open(transcript_path).read()
                question = (
                    SAMPLED_BASELINE_MINERVA_PROMPT.replace("<<<question>>>", question)
                    .replace("<<<answer choices>>>", choices)
                )
                if USE_ASR:
                    question = question.replace("<<<asr>>>", transcript)
            else:
                question = (
                    v["question"]
                    + "\n"
                    + "Answer from the given options: \n"
                    + f"(A) {v['answer_choice_0']}\n"
                    + f"(B) {v['answer_choice_1']}\n"
                    + f"(C) {v['answer_choice_2']}\n"
                    + f"(D) {v['answer_choice_3']}\n"
                    + f"(E) {v['answer_choice_4']}\n"
                )

            video_sets.append(
                {
                    "id": v["key"],
                    "path": video_path,
                    "question": question if not self.is_base_molmo2 else (question, "video_eval_multiple_choice"),
                    "answer": (
                        ANSWER_ID_TO_LETTER[v["answer_id"]]
                        if "sage" in self.model_name
                        else v["answer_id"]
                    ),
                    "full_answer": v[f"answer_choice_{v['answer_id']}"],
                    "video_duration": video_duration,
                }
            )

        # Skip already processed videos
        if os.path.exists(self.output_file):
            with open(self.output_file, "r") as f:
                existing_ids = [json.loads(line)["id"] for line in f]
            video_sets = [v for v in video_sets if v["id"] not in existing_ids]

        return video_sets
    
    def process_video(self, v: Dict, gpu_idx: int) -> Optional[Dict]:
        """Process a single video with appropriate error handling."""
        max_attempts = 5 if "sage" in self.model_name else 10
        print(f"Processing video {v['id']}: {v['path']} for GPU {gpu_idx}")

        answer = None
        complete_results = None
        video_duration = v.get("video_duration", None)

        if not isinstance(v["question"], str):
            assert len(v["question"]) == 2, "Question must be a tuple of (question, eval_style)"
            v["question"], eval_style = v["question"]
        else:
            eval_style = None

        for attempt in range(max_attempts):
            try:
                with timeout(self.timeout_seconds):
                    if "sage" not in self.model_name and ("gemini" in self.model_name or "gpt" in self.model_name or "longrl" in self.model_name or "video-thinker" in self.model_name or "video-r1" in self.model_name):
                        answer = self.model.answer(
                                v["path"],
                                v["question"],
                                model_name=self.model_name,
                                num_sampled_frames=self.num_sampled_frames,
                            )
                        complete_results = None
                    elif "sage" in self.model_name:
                        if attempt > 0:
                            retry_temperature = 0.7
                        else:
                            retry_temperature =0.0
                        answer, complete_results = self.model.run_inference(
                                v["path"],
                                v["question"],
                                model_name=self.model_name,
                                num_sampled_frames=self.num_sampled_frames,
                                temperature=retry_temperature,
                            )
                    else:
                        answer = self.model.context_vlm.client.get_response(
                                prompt=v["question"],
                                media=[v["path"]],
                                media_type="video",
                                eval_style=eval_style
                            )
                        complete_results = None
                    break
            except TimeoutError as e:
                print(f"Timeout on attempt {attempt + 1}/{max_attempts}: {e}")
                if attempt >= max_attempts - 1:
                    answer = f"Could not produce an answer after {max_attempts} attempts due to timeout"
                    complete_results = None
                    break
            except Exception as e:
                if "FileStorageBytesPerProject" in str(e):
                    self.model.context_vlm.delete_client_files()
                print(f"Error on attempt {attempt + 1}/{max_attempts}: {e}")
                
                if attempt >= max_attempts - 1:
                    import traceback
                    tb_str = traceback.format_exc()
                    print(f"Traceback: {tb_str}")
                    print(f"Error: {e}, giving up...")
                    answer = f"Could not produce an answer after {max_attempts} attempts with error {e}"
                    complete_results = None
                    break

        print(f"Answer for video {v['id']} on GPU {gpu_idx}: {answer}")

        result = {
            "path": v["path"],
            "question": v["question"],
            "answer": answer,
            "correct_answer": v["answer"],
            "id": v["id"],
            "full_answer": v["full_answer"],
            "ques_type": v.get("ques_type", None),
            "difficulty": v.get("difficulty", None),
            "modality": v.get("modality", None),
            "num_attempts": attempt + 1,
        }

        if complete_results is not None:
            result["complete_results"] = complete_results

        with open(self.output_file, "a") as f:
            f.write(json.dumps(result) + "\n")
        print(f"Wrote result for video {v['id']}")

        return result

    def process_all_videos(self, max_workers: int = 8, num_gpus: int = 1, gpu_idx: int = 0):
        """Process all videos in parallel."""
        video_sets = self.load_videos()
        video_sets = get_chunk(video_sets, num_gpus, gpu_idx)
        self.video_sets = video_sets

        start_time = time.time()
        
        # Create progress bar for video processing
        progress_bar = tqdm(
            total=len(video_sets),
            desc=f"Processing videos on GPU {gpu_idx}",
            unit="video",
            position=gpu_idx,
            leave=True
        )
        
        for video in video_sets:
            check_tool_call_model()
            self.process_video(video, gpu_idx)
            # Update progress bar
            self.entry += 1
            progress_bar.update(1)
            progress_bar.set_postfix({
                'current_video': video['id'],
                'elapsed': f"{time.time() - start_time:.1f}s"
            })
            print(f"let's move onto {self.entry}")
        
        # Close progress bar
        progress_bar.close()
        
        print(f"Loop completed for {self.model_name} on GPU {self.gpu_idx}")
        if "sage" not in self.model_name and "gemini" in self.model_name:
            self.model.delete_client_files()
        elif "gemini" in self.model_name:
            self.model.context_vlm.delete_client_files()

        print(f"Cleaning up for {self.model_name} on GPU {self.gpu_idx}")
        if "qwen" in self.model_name or "molmo2" in self.model_name:
            self.model.context_vlm.client.cleanup()
        elif "longrl" in self.model_name or "video-thinker" in self.model_name:
            self.model.client.cleanup()
        print(f"returning results for {self.model_name} on GPU {self.gpu_idx}")
        return


def main():
    parser = argparse.ArgumentParser(description="Process Minerva videos with either Gemini or sage")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="minerva",
        help="Benchmark to use for processing (default: minerva)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gemini:gemini-2.5-flash",
        help="Model name to use for processing (default: gemini:gemini-2.5-flash)",
    )
    parser.add_argument(
        "--use_video",
        type=str,
        default="True",
        help="Whether to use video as the media (default: True)",
    )
    parser.add_argument(
        "--use_gemini_as_tool",
        type=str,
        default="False",
        help="Whether to use Gemini as the tool (default: False)",
    )
    parser.add_argument(
        "--num_sampled_frames",
        type=int,
        default=128,
        help="Number of sampled frames (default: 128)",
    )
    parser.add_argument("--workers", type=int, default=1, help="Number of worker threads (default: 8)")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use (default: 1)")
    parser.add_argument("--gpu_idx", type=int, default=0, help="GPU index to use (default: 0)")
    parser.add_argument("--timeout_seconds", type=int, default=300, help="Timeout in seconds for video processing (default: 300)")
    parser.add_argument("--tool_to_drop", type=str, default="None", help="Tool to drop from the model (default: None)")
    parser.add_argument("--max_num_iterative_reasoner_calls", type=int, default=10, help="Maximum number of search LLM (tool) calls (default: 10)")
    args = parser.parse_args()

    processor = Processor(
        model_name=args.model_name,
        num_sampled_frames=args.num_sampled_frames,
        gpu_idx=args.gpu_idx,
        benchmark=args.benchmark,
        timeout_seconds=args.timeout_seconds,
        tool_to_drop=args.tool_to_drop,
        max_num_iterative_reasoner_calls=args.max_num_iterative_reasoner_calls,
        use_gemini_as_tool=args.use_gemini_as_tool == "True",
        use_video=args.use_video == "True",
    )
    processor.process_all_videos(max_workers=args.workers, num_gpus=args.num_gpus, gpu_idx=args.gpu_idx)
    print(f"Done processing for {args.model_name} on GPU {args.gpu_idx}")


if __name__ == "__main__":
    main()