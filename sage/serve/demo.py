import gradio as gr
import json
import os
import shutil
import hashlib
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image, ImageDraw, ImageFont

from sage.main import SAGE
from sage.src.functions.utils.temporal import seconds_to_timestamp, get_video_duration
from sage.src.models.qwen_vl.vision_process import fetch_video as qwen_fetch_video
from sage.src.models.molmo2.vision_process import fetch_video as molmo_fetch_video


def overlay_timestamp(frame: Image.Image, timestamp: str) -> Image.Image:
    """Overlay timestamp text on a frame image."""
    # Create a copy to avoid modifying the original
    frame_with_timestamp = frame.copy()
    draw = ImageDraw.Draw(frame_with_timestamp)
    
    # Try to load a font, fall back to default if not available
    font_size = max(20, min(frame.width, frame.height) // 30)
    font = None
    
    # Try different font paths for cross-platform compatibility
    font_paths = [
        "/System/Library/Fonts/Helvetica.ttc",  # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
        "C:/Windows/Fonts/arial.ttf",  # Windows
    ]
    
    for font_path in font_paths:
        try:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
                break
        except:
            continue
    
    # Fall back to default font if no system font found
    if font is None:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    # Format timestamp text
    text = timestamp
    
    # Calculate text position (top-left corner with padding)
    padding = 10
    x = padding
    y = padding
    
    # Draw a semi-transparent background rectangle for better visibility
    bbox = draw.textbbox((x, y), text, font=font) if font else draw.textbbox((x, y), text)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Draw background rectangle
    overlay = Image.new('RGBA', frame_with_timestamp.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(
        [(x - 5, y - 5), (x + text_width + 5, y + text_height + 5)],
        fill=(0, 0, 0, 180)  # Semi-transparent black background
    )
    frame_with_timestamp = Image.alpha_composite(
        frame_with_timestamp.convert('RGBA'),
        overlay
    ).convert('RGB')
    
    # Redraw text on the composited image
    draw = ImageDraw.Draw(frame_with_timestamp)
    draw.text((x, y), text, fill=(255, 255, 255), font=font)  # White text
    
    return frame_with_timestamp


def extract_sampled_frames_qwen(video_path: str, num_frames: int = 128, video_duration: float = None) -> List[Image.Image]:
    """Use Qwen-VL's video fetching logic to get uniformly sampled frames for display."""
    if not os.path.exists(video_path):
        return []

    # Configure Qwen-VL video element; `nframes` controls how many frames are sampled.
    ele = {
        "type": "video",
        "video": video_path,
        "nframes": num_frames,
    }

    try:
        # Use Qwen-VL's fetch_video to decode and sample frames
        (video_tensor, _video_metadata), _sample_fps = qwen_fetch_video(
            ele,
            return_video_sample_fps=True,
            return_video_metadata=True,
            use_cache=False,
        )
    except Exception as e:
        print(f"Error extracting frames with Qwen-VL fetch_video: {e}")
        return []

    # video_tensor: (T, C, H, W) -> convert each frame to PIL.Image for Gradio Gallery
    frames: List[Image.Image] = []
    try:
        # Get video duration if not provided
        if video_duration is None:
            try:
                video_duration = get_video_duration(video_path)
            except Exception as e:
                print(f"Warning: Could not get video duration: {e}")
                video_duration = 0
        
        num_extracted_frames = len(video_tensor)
        for idx, frame in enumerate(video_tensor):
            # Ensure byte range and convert to HWC numpy
            arr = frame.detach().cpu().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
            frame_img = Image.fromarray(arr)
            
            # Calculate timestamp for this frame
            if video_duration > 0 and num_extracted_frames > 0:
                # Calculate timestamp based on frame index
                timestamp_seconds = (idx / num_extracted_frames) * video_duration
                timestamp_str = seconds_to_timestamp(int(timestamp_seconds), in_hr=video_duration >= 3600)
                frame_img = overlay_timestamp(frame_img, timestamp_str)
            
            frames.append(frame_img)
    except Exception as e:
        print(f"Error converting sampled frames to images: {e}")
        return []

    return frames


def extract_sampled_frames_molmo(video_path: str, num_frames: int = 128, video_duration: float = None) -> List[Image.Image]:
    """Use Molmo2's video fetching logic to get uniformly sampled frames for display."""
    if not os.path.exists(video_path):
        return []

    # Molmo2's fetch_video expects a config dict; `max_frames` controls the cap.
    ele = {
        "video": video_path,
        "max_frames": num_frames,
        # Use the default frame_sample_mode ("uniform_last_frame") so it matches model preprocessing.
    }

    try:
        video_info = molmo_fetch_video(ele, use_cache=False)
    except Exception as e:
        print(f"Error extracting frames with Molmo2 fetch_video: {e}")
        return []

    frames_array = video_info.get("frames", None)
    if frames_array is None:
        return []

    frames: List[Image.Image] = []
    try:
        # Get video duration if not provided
        if video_duration is None:
            try:
                video_duration = get_video_duration(video_path)
            except Exception as e:
                print(f"Warning: Could not get video duration: {e}")
                video_duration = 0
        
        num_extracted_frames = len(frames_array)
        # frames_array: (T, H, W, C) in RGB
        for idx, frame in enumerate(frames_array):
            frame_img = Image.fromarray(frame.astype("uint8"))
            
            # Calculate timestamp for this frame
            if video_duration > 0 and num_extracted_frames > 0:
                # Calculate timestamp based on frame index
                timestamp_seconds = (idx / num_extracted_frames) * video_duration
                timestamp_str = seconds_to_timestamp(int(timestamp_seconds), in_hr=video_duration >= 3600)
                frame_img = overlay_timestamp(frame_img, timestamp_str)
            
            frames.append(frame_img)
    except Exception as e:
        print(f"Error converting Molmo2 sampled frames to images: {e}")
        return []

    return frames


class MolmoDemo:
    def __init__(self, examples_dir="examples"):
        """
        Args:
            examples_dir: Directory containing example video files
        """
        # Always work with an absolute examples directory and make sure it exists
        self.examples_dir = os.path.abspath(examples_dir)
        os.makedirs(self.examples_dir, exist_ok=True)
        
        # Load videos from examples directory
        self.example_videos = self._load_videos_from_examples(self.examples_dir)
        print(f"Loaded {len(self.example_videos)} example videos from {examples_dir}")

        self.current_video_path = None
        self.video_mode = "upload"  # "upload" or "example"
        self.sampled_frames = []  # Store extracted frames for display

        self.state = {
            "prior_context": [],
            "tool_info": [],
            "results": [],
            "final_answer": "",
        }
        self.tools_so_far = {}

    def _load_videos_from_examples(self, examples_dir):
        """Load videos from examples directory using os.listdir.
        
        Args:
            examples_dir: Directory containing video files
        
        Returns:
            List of video file paths
        """
        videos = []
        examples_dir = os.path.abspath(examples_dir)
        
        if not os.path.isdir(examples_dir):
            print(f"Warning: Examples directory '{examples_dir}' does not exist or is not a directory")
            return videos
        
        # Supported video extensions
        video_extensions = (".mp4", ".MP4", ".mkv", ".avi", ".mov", ".webm")
        
        try:
            for fname in os.listdir(examples_dir):
                fpath = os.path.join(examples_dir, fname)
                if os.path.isfile(fpath) and fname.endswith(video_extensions):
                    videos.append(os.path.abspath(fpath))
        except OSError as e:
            print(f"Error reading examples directory '{examples_dir}': {e}")
            return []
        
        return sorted(videos)

    def _compute_file_hash(self, file_path: str) -> Optional[str]:
        """Compute SHA256 hash of a file."""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                # Read file in chunks to handle large files efficiently
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            print(f"Error computing hash for {file_path}: {e}")
            return None

    def _find_existing_video_by_hash(self, target_hash: str) -> Optional[str]:
        """Find an existing video in examples directory with the same hash."""
        if not target_hash:
            return None
        
        video_extensions = (".mp4", ".MP4", ".mkv", ".avi", ".mov", ".webm")
        
        try:
            for fname in os.listdir(self.examples_dir):
                fpath = os.path.join(self.examples_dir, fname)
                if os.path.isfile(fpath) and fname.endswith(video_extensions):
                    existing_hash = self._compute_file_hash(fpath)
                    if existing_hash == target_hash:
                        return os.path.abspath(fpath)
        except OSError as e:
            print(f"Error reading examples directory: {e}")
        
        return None

    def _save_uploaded_video_to_examples(self, src_path: str) -> str:
        """Save an uploaded video into the examples directory and return its new path."""
        if not src_path or not os.path.exists(src_path):
            return src_path

        src_path_abs = os.path.abspath(src_path)

        # If it's already inside the examples directory, just return it
        try:
            if os.path.commonpath([self.examples_dir, src_path_abs]) == self.examples_dir:
                return src_path_abs
        except ValueError:
            # In case paths are on different drives or invalid, fall back to hash check
            pass

        # Compute hash of source video to check for duplicates
        src_hash = self._compute_file_hash(src_path_abs)
        if src_hash:
            # Check if a video with the same content already exists
            existing_video = self._find_existing_video_by_hash(src_hash)
            if existing_video:
                print(f"Video with same content already exists: {existing_video}")
                return existing_video

        base_name = os.path.basename(src_path_abs)
        dst_path = os.path.join(self.examples_dir, base_name)

        # Avoid overwriting an existing file by adding a numeric suffix if needed
        if os.path.exists(dst_path):
            name, ext = os.path.splitext(base_name)
            counter = 1
            while True:
                candidate = f"{name}_{counter}{ext}"
                candidate_path = os.path.join(self.examples_dir, candidate)
                if not os.path.exists(candidate_path):
                    dst_path = candidate_path
                    break
                counter += 1

        try:
            shutil.copy2(src_path_abs, dst_path)
        except Exception as e:
            print(f"Failed to copy uploaded video to examples directory: {e}")
            return src_path_abs

        return os.path.abspath(dst_path)



    def handle_video_change(self, video_file):
        """Handle video file change (upload or selection)"""
        if video_file is None:
            return {
                self.sampled_frames_display: [],
            }
        
        # Handle different file input types from Video component
        if isinstance(video_file, str):
            raw_video_path = video_file
        elif isinstance(video_file, dict) and 'name' in video_file:
            raw_video_path = video_file['name']
        elif hasattr(video_file, 'name'):
            raw_video_path = video_file.name
        else:
            raw_video_path = str(video_file)

        # Normalize the path to check if it's from examples
        raw_video_path_abs = os.path.abspath(raw_video_path)
        
        # Check if this video is already in the examples directory
        # If it is, use it directly without saving/copying
        is_example_video = raw_video_path_abs in [os.path.abspath(v) for v in self.example_videos]
        
        if is_example_video:
            # Video is from examples, use it directly without saving
            video_path = raw_video_path_abs
        else:
            # New upload, save it to examples directory
            video_path = self._save_uploaded_video_to_examples(raw_video_path)

        # This is an uploaded or selected video
        self.current_video_path = video_path
        self.video_mode = "upload"
        
        # Extract frames for display
        if os.path.exists(video_path):
            # Get video duration for timestamp calculation
            try:
                video_duration = get_video_duration(video_path)
            except Exception as e:
                print(f"Warning: Could not get video duration: {e}")
                video_duration = None
            
            # Choose frame sampling backend based on environment for parity with underlying model.
            if os.environ.get("IS_MOLMO2", "false").lower() in {"1", "true", "yes"}:
                self.sampled_frames = extract_sampled_frames_molmo(video_path, num_frames=128, video_duration=video_duration)
            else:
                self.sampled_frames = extract_sampled_frames_qwen(video_path, num_frames=128, video_duration=video_duration)
        else:
            self.sampled_frames = []
        
        return {
            self.sampled_frames_display: self.sampled_frames,
        }

    def format_results(self):
        return (
            self.state["prior_context"],
            self.state["tool_info"],
            self.state["results"],
            self.state.get("final_answer", ""),
        )

    def clear_everything(self, video_display_component):
        """Clear all video, query, and results"""
        self.current_video_path = None
        self.video_mode = "upload"
        self.sampled_frames = []
        self.state = {
            "prior_context": [],
            "tool_info": [],
            "results": [],
            "final_answer": "",
        }
        return {
            video_display_component: None,
            self.query_input: "",
            self.context_vlm_display: [],
            self.function_call_display: [],
            self.iterative_reasoner_display: [],
            self.sampled_frames_display: [],
            self.final_answer_display: "",
        }

    def process_video(self, query):
        model_name = self.model_name

        self.state["prior_context"] = ["Processing video..."]
        self.state["tool_info"] = ["Processing video..."]
        self.state["results"] = ["Processing video..."]
        self.state["final_answer"] = "Processing video..."
        yield self.format_results()

        tools_so_far = {}
        self.sage.num_tool_calls = {}

        # Get video path
        if self.current_video_path is None:
            self.state["prior_context"] = ["Error: No video selected"]
            self.state["final_answer"] = "Error: No video selected"
            yield self.format_results()
            return
        
        video_path = self.current_video_path
        if not os.path.exists(video_path):
            self.state["prior_context"] = [f"Error: Video file not found: {video_path}"]
            self.state["final_answer"] = f"Error: Video file not found: {video_path}"
            yield self.format_results()
            return
        
        video_duration = self.sage.context_vlm.get_video_duration(video_path)
        timestamp_format = seconds_to_timestamp(video_duration, in_hr=True)

        # Step 1: Context VLM
        context_result = self.sage.get_context(
            video_path=video_path,
            query=query,
            model_name=model_name,
            sample_frames=True,
            num_sampled_frames=128,
            return_ids=False,
        )

        # Normalize and display prior context
        prior_context = context_result.get("vlm_response", {})
        if not isinstance(prior_context, dict):
            # Non-JSON response; show as-is and stop
            self.state["prior_context"] = prior_context
            self.state["tool_info"] = []
            self.state["results"] = []
            if isinstance(prior_context, str):
                self.state["final_answer"] = prior_context
            yield self.format_results()
            return

        final_answer = prior_context.get("final_answer", None)
        if not isinstance(prior_context.get("recommended_tools", {}), dict):
            prior_context["recommended_tools"] = {}

        self.state["prior_context"] = prior_context
        self.state["results"] = []
        self.state["tool_info"] = []
        if final_answer is not None:
            self.state["final_answer"] = final_answer
        yield self.format_results()

        # If tools are recommended, execute them and iterate with iterative reasoner
        if bool(prior_context.get("recommended_tools", {}).get("needed", False)) and len(prior_context.get("recommended_tools", {}).get("tool_calls", [])) > 0:
            tool_calls_result = self.sage.get_tool_calls(prior_context)

            # Track args validity across calls
            args_valid = True
            for tc in tool_calls_result.get("tool_calls", {}).values():
                args_valid = args_valid and tc.get("args_validity", True)

            if tool_calls_result is not None:
                tools_so_far = {**tools_so_far, **tool_calls_result.get("tool_calls", {})}

            # If using Gemini and arguments are invalid, stop early (to match run_inference)
            if "gemini" in model_name.lower() and not args_valid:
                self.state["tool_info"] = tools_so_far
                yield self.format_results()
                return

            self.state["tool_info"] = tools_so_far
            yield self.format_results()

            # Step 2: Iterate with Iterative Reasoner
            call_iterative_reasoner = True
            num_search_calls = 0
            max_calls = self.sage.max_num_iterative_reasoner_calls
            max_tools_so_far = int(os.environ.get("MAX_TOOLS_SO_FAR", "10"))

            while call_iterative_reasoner:
                num_search_calls += 1
                if max_calls > 0 and num_search_calls > max_calls:
                    final_answer = f"Could not produce an answer after {max_calls} iterative reasoner calls"
                    self.state["final_answer"] = final_answer
                    break

                iterative_reasoner_result = self.sage.get_iterative_reasoner_results(
                    query=query,
                    video_path=video_path,
                    model_name=model_name,
                    results=tool_calls_result,
                    tools_so_far=self.sage.limit_tools_so_far(tools_so_far, max_tools=max_tools_so_far),
                    visual_context=prior_context.get("video_context", ""),
                    timestamp_format=timestamp_format,
                    video_duration=video_duration,
                    return_ids=False,
                )
                self.state["results"].append(iterative_reasoner_result)
                yield self.format_results()

                iterative_reasoner_results = iterative_reasoner_result.get("iterative_reasoner_results", {})
                if not isinstance(iterative_reasoner_results, dict):
                    # Non-JSON response; show and stop
                    return

                final_answer = iterative_reasoner_results.get("final_answer", None)
                if not isinstance(iterative_reasoner_results.get("answerable", {}), dict):
                    iterative_reasoner_results["answerable"] = {}

                if not bool(iterative_reasoner_results.get("answerable", {}).get("verdict", False)):
                    call_iterative_reasoner = True

                    if iterative_reasoner_results.get("recommended_tools", {}) is None:
                        iterative_reasoner_results["recommended_tools"] = {}

                    if bool(iterative_reasoner_results.get("recommended_tools", {}).get("needed", False)):
                        tool_calls_result = self.sage.get_tool_calls(iterative_reasoner_results)
                        tools_so_far = {**tools_so_far, **tool_calls_result.get("tool_calls", {})}

                        # Update args validity and apply Gemini early stop if invalid
                        args_valid = True
                        for tc in tool_calls_result.get("tool_calls", {}).values():
                            args_valid = args_valid and tc.get("args_validity", True)
                        self.state["tool_info"] = tools_so_far
                        yield self.format_results()

                        if "gemini" in model_name.lower() and not args_valid:
                            return
                    else:
                        tool_calls_result = {
                            "tool_calls": {
                                "None": {
                                    "result": "No tool calls found but a final answer was not returned.",
                                    "arguments": {},
                                    "args_validity": False,
                                    "rationale": "No rationale found",
                                }
                            }
                        }
                        tools_so_far = {**tools_so_far, **tool_calls_result.get("tool_calls", {})}
                        self.state["tool_info"] = tools_so_far
                        yield self.format_results()

                        if "gemini" in model_name.lower():
                            return
                elif final_answer is None:
                    call_iterative_reasoner = True
                    tool_calls_result = {
                        "tool_calls": {
                            "None": {
                                "result": "No tool calls found but a final answer was not returned.",
                                "arguments": {},
                                "args_validity": False,
                                "rationale": "No rationale found",
                            }
                        }
                    }
                    tools_so_far = {**tools_so_far, **tool_calls_result.get("tool_calls", {})}
                    self.state["tool_info"] = tools_so_far
                    yield self.format_results()

                    if "gemini" in model_name.lower():
                        return
                else:
                    # Got a final answer marked answerable
                    call_iterative_reasoner = False
                    self.state["final_answer"] = final_answer if final_answer is not None else ""

        # Final yield of state
        yield self.format_results()

    def create_interface(self, model_name):
        self.sage = SAGE(vlm_api_type=model_name)
        self.model_name = model_name
        with gr.Blocks(
            css="""
            body {
                background-color: #1e1e1e;
            }
            .gradio-container {
                max-width: none;
                width: 100vw
            }
            .card {
                background-color: #2a2a2a;
                padding: 20px;
                border-radius: 12px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.3);
                height: 100%;
            }
            .text-center {
                text-align: center;
            }
        """
        ) as demo:
            gr.Markdown("## 🎥 SAGE Video Analysis Demo", elem_classes=["text-center"])

            with gr.Row(equal_height=True, scale=10):
                # Left Panel: Video Browser
                with gr.Column(scale=5, elem_classes=["card"]):
                    gr.Markdown("### 📂 Video Browser")
                    
                    video_display = gr.Video(
                        label="Video (Upload or select from examples below)",
                    )

                # Right Panel: Video Analysis
                with gr.Column(scale=5, elem_classes=["card"]):
                    gr.Markdown("### 🧠 Video Analysis")

                    self.query_input = gr.Textbox(
                        label="Analysis Query",
                        placeholder="Ask a question about the video...",
                        value="",
                        lines=3,
                    )

                    analyze_btn = gr.Button("🚀 Analyze Video", interactive=False)

                    with gr.Row():
                        clear_btn = gr.Button("Clear", variant="secondary")
                        stop_btn = gr.Button("⛔️ Stop", variant="stop")

                    # Dedicated final answer display box
                    self.final_answer_display = gr.Textbox(
                        label="Final Answer",
                        interactive=False,
                        show_label=True,
                    )
            
            with gr.Accordion("📸 Sampled 128 Frames", open=False):
                self.sampled_frames_display = gr.Gallery(
                    show_label=True,
                    elem_id="sampled_frames_gallery",
                    columns=8,
                    rows=16,
                    height="auto",
                    object_fit="contain"
                )

            with gr.Accordion("Context VLM Response", open=True):
                self.context_vlm_display = gr.JSON(label="Context VLM Response")

            with gr.Accordion("Tool Call Results", open=False):
                self.function_call_display = gr.JSON(label="Tool Call Results")

            with gr.Accordion("Iterative Reasoner Results", open=True):
                self.iterative_reasoner_display = gr.JSON(label="Iterative Reasoner Results")
            
            # Examples section at bottom
            if self.example_videos:
                gr.Markdown("---")
                gr.Markdown("### 📁 Example Videos")
                gr.Markdown(
                    "Click an example below to load it into the video player. "
                    "You can then modify the query and run analysis."
                )

                # Use Gradio Examples to load example videos directly into the Video component.
                # Each example is a single-column list whose value is the absolute video path.
                example_data = [[video_path] for video_path in self.example_videos]
                gr.Examples(
                    examples=example_data,
                    inputs=[video_display],
                    label="Example Videos",
                    examples_per_page=len(example_data),
                )
            
            # Helper function to enable/disable analyze button based on query
            def update_analyze_button(query):
                return gr.update(interactive=bool(query and query.strip()))
            
            # Handle video change (upload or selection)
            video_display.change(
                fn=self.handle_video_change,
                inputs=[video_display],
                outputs=[
                    self.sampled_frames_display,
                ],
            )
            
            # Update analyze button state when query changes
            self.query_input.change(
                fn=update_analyze_button,
                inputs=[self.query_input],
                outputs=[analyze_btn],
            )

            # Handle clear button
            def clear_all(video_display_component):
                result = self.clear_everything(video_display_component)
                # Return as tuple matching the outputs order, with button disabled
                return (
                    result[video_display_component],
                    result[self.query_input],
                    result[self.context_vlm_display],
                    result[self.function_call_display],
                    result[self.iterative_reasoner_display],
                    result[self.sampled_frames_display],
                    result[self.final_answer_display],
                    gr.update(interactive=False),  # Disable analyze button
                )
            
            clear_btn.click(
                fn=clear_all,
                inputs=[video_display],
                outputs=[
                    video_display,
                    self.query_input,
                    self.context_vlm_display,
                    self.function_call_display,
                    self.iterative_reasoner_display,
                    self.sampled_frames_display,
                    self.final_answer_display,
                    analyze_btn,
                ],
            )

            click_event = analyze_btn.click(
                self.process_video,
                inputs=[
                    # iterative_reasoner_dropdown,
                    self.query_input,
                ],
                outputs=[
                    self.context_vlm_display,
                    self.function_call_display,
                    self.iterative_reasoner_display,
                    self.final_answer_display,
                ],
            )

            # Also trigger on Enter key press in the query input
            submit_event = self.query_input.submit(
                self.process_video,
                inputs=[
                    self.query_input,
                ],
                outputs=[
                    self.context_vlm_display,
                    self.function_call_display,
                    self.iterative_reasoner_display,
                    self.final_answer_display,
                ],
            )

            stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[click_event, submit_event])

        return demo


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SAGE Video Analysis Demo")
    parser.add_argument(
        "model_name",
        type=str,
        help="Model name for VLM API type"
    )
    parser.add_argument(
        "--examples-dir",
        type=str,
        default="sage/serve/examples",
        help="Directory containing example video files (default: examples)"
    )
    args = parser.parse_args()
    
    demo = MolmoDemo(examples_dir=args.examples_dir)
    interface = demo.create_interface(model_name=args.model_name)
    
    # Collect allowed paths for Gradio
    allowed_paths = []
    if demo.examples_dir:
        examples_path = Path(demo.examples_dir).absolute()
        if examples_path.exists():
            allowed_paths.append(str(examples_path))
    
    interface.launch(allowed_paths=allowed_paths if allowed_paths else None)
