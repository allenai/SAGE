import os
import re
import requests
import json
import base64
import asyncio
from termcolor import colored
from typing import Optional, List, Dict, Any, Tuple
from sage.src.context_vlm import ContextVLM
from sage.utils.utils import (
    ITERATIVE_REASONER_MSG,
    ITERATIVE_REASONER_PROMPT,
    SAGE_ITERATIVE_REASONER_PROMPT,
    SAGE_ITERATIVE_REASONER_MSG_PROMPT,
)
from sage.src.functions.utils.utils import get_functions_from_folder, format_tool_descs
from sage.src.api.response import get_response
from sage.src.functions.utils.temporal import seconds_to_timestamp, timestamp_to_seconds, fix_timestamp

VIDEO_DIR = os.environ.get("VIDEO_DIR", "None")
if VIDEO_DIR == "None":
    raise ValueError("VIDEO_DIR is not set")

class SAGE:
    """
    Main system that integrates the Context VLM with various video analysis tools.
    """

    def __init__(self, 
            vlm_api_type: str = None, vllm_engine: object = None, 
            processor: object = None, drop_tool_call_files: list = [], 
            max_num_iterative_reasoner_calls: int = 10,
            is_rl_train_mode: bool = False, 
            use_gemini_as_tool: bool = False,
            use_video: bool = True, tool_call_clients: object = None,
            num_tool_calls: dict = None,
        ):

        self.context_vlm = ContextVLM(
            api_type=vlm_api_type, vllm_engine=vllm_engine, 
            processor=processor, 
            drop_tool_call_files=drop_tool_call_files, 
            is_rl_train_mode=is_rl_train_mode,
            use_video=use_video,
            tool_call_clients=tool_call_clients,
        )
        self.drop_tool_call_files = drop_tool_call_files
        self.tools, self.dispatcher = get_functions_from_folder("sage/src/functions/tools", drop_tool_call_files=drop_tool_call_files)
        self.max_num_iterative_reasoner_calls = max_num_iterative_reasoner_calls
        self.use_gemini_as_tool = use_gemini_as_tool
        self.tool_call_clients = tool_call_clients
        self.is_rl_train_mode = is_rl_train_mode
        self.num_tool_calls = num_tool_calls if num_tool_calls is not None else {}
        self.use_video = use_video
        if not is_rl_train_mode:
            print(f"max_num_iterative_reasoner_calls: {self.max_num_iterative_reasoner_calls}")
        assert len(self.tools) > 0, "No tools found"

    def format_tool_calls(self, tool_calls: List[Dict[str, Any]]):
        tool_calls_str = ""
        for tool_call in tool_calls:
            tool_calls_str += f"{tool_call} output: {tool_calls[tool_call].get('result', 'No result found')}\n"
        return tool_calls_str

    def format_tool_calls_args(self, tool_calls: Dict[str, Any]):
        tool_calls_str_args = ""
        for tool_call in tool_calls:
            tool_calls_str_args += f"{tool_call} with arguments {tool_calls[tool_call].get('arguments', {})}\n"
        return tool_calls_str_args

    def limit_tools_so_far(self, tools_so_far: Dict[str, Any], max_tools: int = 5) -> Dict[str, Any]:
        """Limit tools_so_far to only the most recent max_tools entries."""
        if len(tools_so_far) <= max_tools:
            return tools_so_far
        
        # Convert to list of tuples and sort by key (assuming keys contain sequence numbers)
        tools_list = list(tools_so_far.items())
        
        # Sort by the tool name to maintain order (assuming format like "tool_name_#number")
        tools_list.sort(key=lambda x: x[0])
        
        # Take only the most recent max_tools
        limited_tools = dict(tools_list[-max_tools:])
        return limited_tools

    def get_context_vlm_prompt(
        self,
        query: str,
        video_path: str,
        sample_frames: bool = False,
        num_sampled_frames: int = 128,
    ) -> str:
        return self.context_vlm.get_context_vlm_prompt(
            query,
            video_path,
            self.tools,
            sample_frames,
            num_sampled_frames,
        )[0]

    def get_context(
        self,
        video_path: str,
        query: str,
        model_name: str,
        sample_frames: bool = False,
        use_transcript: bool = False,
        num_sampled_frames: int = 128,
        return_ids: bool = False,
        temperature: float = None,
        **kwargs,
    ) -> Dict[str, Any]:

        # Get VLM analysis
        vlm_response = self.context_vlm.analyze(
            video_path,
            query,
            model_name=model_name,
            tools=self.tools,
            sample_frames=sample_frames,
            use_transcript=use_transcript,
            num_sampled_frames=num_sampled_frames,
            return_ids=return_ids,
            temperature=temperature,
            **kwargs,
        )

        return vlm_response

    def get_tool_calls(self, vlm_response: str, **kwargs) -> List[Dict[str, Any]]:
        tool_calls = self.parse_tool_call(vlm_response.get("recommended_tools", {}).get("tool_calls", []))
        if len(tool_calls) > 0:
            results = {"tool_calls": self.execute_tool_call(tool_calls, **kwargs)}
            return results
        else:
            if "None" not in self.num_tool_calls:
                self.num_tool_calls["None"] = 0
            self.num_tool_calls["None"] += 1
            return {"tool_calls": {f"None_#{self.num_tool_calls.get('None')}": {"result": "No tool calls found", "arguments": {}, "args_validity": False}}}

    def get_iterative_reasoner_results(
        self,
        query: str,
        video_path: str,
        results: Dict[str, Any] = None,
        model_name: str = "gemini:gemini-2.5-flash",
        tools_so_far: List[str] = [],
        visual_context: str = "None",
        video_duration: str = None,
        timestamp_format: str = None,
        return_ids: bool = False,
        temperature: float = None,
        **kwargs,
    ) -> Dict[str, Any]:

        if not self.use_video:
            timestamp_format = "video duration not available"
            video_path = "video not provided"
            video_duration = "video duration not available"

        if "qwen" in model_name.lower() or "molmo2" in model_name.lower():
            video_info_prompt = f"Video Path: {video_path} of duration {timestamp_format}."
            prev_tool_call = self.format_tool_calls_args(results["tool_calls"])
            query = (
                SAGE_ITERATIVE_REASONER_MSG_PROMPT.
                    replace("<<<query>>>", query).
                    replace("<<<video_info>>>", video_info_prompt).
                    replace("<<<visual_context>>>", str(visual_context)).
                    replace("<<<tools_so_far>>>", self.format_tool_calls(tools_so_far)).
                    replace("<<<previous_tool_call>>>", prev_tool_call)
            )
            system_prompt = SAGE_ITERATIVE_REASONER_PROMPT.replace("<<<tools>>>", format_tool_descs(self.tools))
        else:
            query = (
                ITERATIVE_REASONER_MSG.replace("<<<query>>>", query)
                .replace("<<<video_path>>>", video_path)
                .replace("<<<tool_results>>>", self.format_tool_calls(results["tool_calls"]))
                .replace("<<<visual_context>>>", str(visual_context))
                .replace("<<<tools>>>", format_tool_descs(self.tools))
                .replace("<<<tools_so_far>>>", self.format_tool_calls(tools_so_far))
                .replace("<<<timestamp_format>>>", timestamp_format)
                .replace("<<<video_duration>>>", str(video_duration))
            )
            system_prompt = ITERATIVE_REASONER_PROMPT
        
        if return_ids:
            iterative_reasoner_results = get_response(
                message=query,
                sys_prompt=system_prompt,
                model_name=model_name,
                model=self.context_vlm.client,
                return_ids=return_ids,
                temperature=temperature,
                **kwargs,
            )
        else:
            iterative_reasoner_results = self.context_vlm._retry_with_exponential_backoff(
                get_response,
                message=query,
                sys_prompt=system_prompt,
                model_name=model_name,
                model=self.context_vlm.client,
                return_ids=return_ids,
                temperature=temperature,
                **kwargs,
            )
            
        if return_ids:
            iterative_reasoner_results, _, completion_ids, prompt_ids, attention_mask = iterative_reasoner_results
            results["completion_ids"] = completion_ids
            results["prompt_ids"] = prompt_ids
            results["attention_mask"] = attention_mask
        else:
            iterative_reasoner_results, _ = iterative_reasoner_results
        if iterative_reasoner_results is None:
            iterative_reasoner_results = {}

        results["iterative_reasoner_results"] = iterative_reasoner_results
        # results["inputs"] = {
        #     "query": query,
        #     "model_name": model_name,
        # }
        return results

    
    def fix_args(self, args):
        if "video_path" in args and ".mp4" in args["video_path"] and not os.path.exists(args["video_path"]):
            vid_path = os.path.join(VIDEO_DIR, args["video_path"].split("/")[-1])
            if os.path.exists(vid_path):
                args["video_path"] = vid_path
        if "media_paths" in args and isinstance(args["media_paths"], list):
            new_media_paths = []
            for media_path in args["media_paths"]:
                if not os.path.exists(media_path):
                    if ".mp4" in media_path:
                        vid_path = os.path.join(VIDEO_DIR, media_path.split("/")[-1])
                        if os.path.exists(vid_path):
                            new_media_paths.append(vid_path)
                        else:
                            new_media_paths.append(media_path)
                    else:
                        folder = media_path.split("/")[-2]
                        file = media_path.split("/")[-1]
                        vid_path = os.path.join(VIDEO_DIR, folder, file)
                        if os.path.exists(vid_path):
                            new_media_paths.append(vid_path)
                        else:
                            new_media_paths.append(media_path)
            args["media_paths"] = new_media_paths
        return args
    
    def parse_tool_call(self, tool_calls):
        function_calls = []
        if tool_calls is None or not isinstance(tool_calls, list):
            return function_calls
        
        if len(tool_calls) > 1:
            tool_calls = [tool_calls[0]]
        
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            if tool_call.get("name", "None") not in self.dispatcher:
                continue
            if tool_call.get("name", "None") not in self.num_tool_calls:
                self.num_tool_calls[tool_call.get("name", "None")] = 0
            self.num_tool_calls[tool_call.get("name", "None")] += 1
            args = tool_call.get("arguments", {})
            args = self.fix_args(args)
            function_calls.append(
                {
                    "name": f"{tool_call.get('name', 'None')}_#{self.num_tool_calls[tool_call.get('name', 'None')]}",
                    "params": args,
                }
            )

        return function_calls

    def execute_tool_call(self, function_calls, **kwargs):
        results = {}
        for idx, tool_call in enumerate(function_calls):
            tool_name = tool_call["name"].split("_#")[0]
            if tool_name in self.dispatcher:
                function = self.dispatcher.get(tool_name)["function"]
                func_args = self.dispatcher.get(tool_name)["args"]
                func_args_validator = self.dispatcher.get(tool_name)["args_validator"]
                org_arguments = tool_call["params"]
                if isinstance(org_arguments, list):
                    arguments = org_arguments[0]
                if not isinstance(org_arguments, dict):
                    raise ValueError(f"arguments must be a dictionary, found {type(org_arguments)}, {org_arguments} for {tool_call['name']}")
                arguments = {}
                for k, v in org_arguments.items():
                    if k in func_args:
                        arguments[k] = v
                ok, errors = func_args_validator.validate(arguments)
                # print(colored(f"Executing {tool_call['name']} with arguments:\n{{\n  {arguments}\n}}", 'cyan', attrs=['bold']))
                try:
                    if not ok:
                        raise ValueError(f"Invalid arguments for {tool_call['name']}: {errors}")
                    if not self.use_gemini_as_tool and "gemini" not in self.context_vlm.api_type and "gpt" not in self.context_vlm.api_type and tool_name in ["perform_reasoning", "identify_timestamps_visually"]:
                        # call the using self.context_vlm.client
                        method = getattr(self.context_vlm.client, tool_name)
                        function_call_result = method(**arguments, **kwargs)
                    else:
                        function_call_result = function(**arguments)
    
                    results[f"{tool_call['name']}"] = {
                        "result": function_call_result,
                        "arguments": arguments,
                        "rationale": tool_call.get("rationale"),
                        "args_validity": ok,
                    }
                except Exception as e:
                    # import traceback
                    # tb_str = traceback.format_exc()
                    print(colored(
                        f"Error executing {tool_call['name']}: {e} for arguments {arguments}",
                        # \nFull traceback:\n{tb_str}",
                        'red', attrs=['bold']
                    ))
                    results[f"{tool_call['name']}"] = (
                        {
                            "result": f"Error executing {tool_call['name']}: {str(e)}\n for argument {str(arguments)}",
                            "arguments": arguments,
                            "args_validity": ok,
                            "rationale": tool_call.get("rationale"),
                        }
                    )

                # Check all previous tool call results for the next tool call
                if idx < len(function_calls) - 1:
                    next_tool_call = function_calls[idx + 1]
                    # First pass: collect all values for list-type arguments
                    list_args = {}
                    for prev_idx in range(idx + 1):
                        prev_result = results[f"{function_calls[prev_idx]['name']}"]
                        if isinstance(prev_result, dict):
                            for k, v in next_tool_call["params"].items():
                                if k in prev_result:
                                    if isinstance(prev_result[k], list):
                                        if k not in list_args:
                                            list_args[k] = []
                                        list_args[k].extend(prev_result[k])

                    # Second pass: update non-list arguments with most recent value
                    for prev_idx in range(idx, -1, -1):  # Go backwards to get most recent first
                        prev_result = results[f"{function_calls[prev_idx]['name']}"]
                        if isinstance(prev_result, dict):
                            for k, v in next_tool_call["params"].items():
                                if k in prev_result and not isinstance(prev_result[k], list):
                                    next_tool_call["params"][k] = prev_result[k]
                                    # print(colored(f"Next tool call {next_tool_call['name']} argument {k}: {next_tool_call['params'][k]} updated to {prev_result[k]} from {function_calls[prev_idx]['name']}", 'magenta', attrs=['bold']))

                    # Finally, update list arguments with accumulated values
                    for k, v in list_args.items():
                        next_tool_call["params"][k] = v
                        # print(colored(f"Next tool call {next_tool_call['name']} argument {k}: {next_tool_call['params'][k]} updated to accumulated list {v}", 'magenta', attrs=['bold']))
            else:
                results[f"{tool_call['name']}"] = (
                    {
                        "result": f"Error executing: {tool_call['name']} is not a valid tool",
                        "arguments": tool_call["params"],
                        "args_validity": False,
                        "rationale": tool_call.get("rationale"),
                    }
                )
        return results

    def run_inference(
        self,
        video_path: str,
        query: str,
        model_name: str = "gemini:gemini-2.5-flash",
        sample_frames: bool = False,
        num_sampled_frames: int = 128,
        return_ids: bool = False,
        temperature: float = None,
        **kwargs,
    ) -> Tuple[str, Dict]:
        """Get answer using sage model."""
        complete_results = {"context_vlm": [], "iterative_reasoner": [], "num_iterative_reasoner_calls": 0, "args_validity": True}
        tools_so_far = {}
        self.num_tool_calls = {}
        if not self.use_video:
            timestamp_format = "video duration not available"
            video_duration = 0
        else:
            video_duration = self.context_vlm.get_video_duration(video_path)
            timestamp_format = seconds_to_timestamp(video_duration, in_hr=True)

        completion_ids = []
        prompt_ids = []
        attention_mask = []

        # print(f"Getting context for a video of duration {timestamp_format}...")
        context_result = self.get_context(
            video_path=video_path,
            query=query,
            model_name=model_name,
            sample_frames=sample_frames,
            num_sampled_frames=num_sampled_frames,
            return_ids=return_ids,
            temperature=temperature,
            **kwargs,
        )
        
        complete_results["context_vlm"].append(context_result)
        if return_ids:
            completion_ids.extend(context_result["completion_ids"])
            prompt_ids.append((context_result["prompt_ids"], context_result["completion_ids"]))
            attention_mask.append(context_result["attention_mask"])
            if not isinstance(context_result["vlm_response"], dict):
                return context_result["vlm_response"], complete_results, completion_ids, prompt_ids, attention_mask
        
        prior_context = context_result["vlm_response"]
        final_answer = prior_context.get("final_answer", None)

        if not isinstance(context_result.get("vlm_response", {}).get("recommended_tools", {}), dict):
            context_result["vlm_response"]["recommended_tools"] = {}

        if bool(context_result.get("vlm_response", {}).get("recommended_tools", {}).get("needed", False)) and len(context_result.get("vlm_response", {}).get("recommended_tools", {}).get("tool_calls", [])) > 0:
            tool_calls_result = self.get_tool_calls(context_result.get("vlm_response", {}), **kwargs)
            for tool_call in tool_calls_result.get("tool_calls", {}).values():
                complete_results["args_validity"] = tool_call.get("args_validity", True)
            
            if tool_calls_result is not None:
                tools_so_far = {**tools_so_far, **tool_calls_result.get("tool_calls", {})}
            
            if return_ids or "gemini" in model_name.lower():
                if not complete_results["args_validity"]:
                    if "gemini" in model_name.lower():
                        return final_answer, complete_results
                    return final_answer, complete_results, completion_ids, prompt_ids, attention_mask

            call_iterative_reasoner = True
            while call_iterative_reasoner:
                complete_results["num_iterative_reasoner_calls"] += 1
                if complete_results["num_iterative_reasoner_calls"] > self.max_num_iterative_reasoner_calls:
                    final_answer = f"Could not produce an answer after {self.max_num_iterative_reasoner_calls} iterative reasoner calls"
                    break
                
                iterative_reasoner_result = self.get_iterative_reasoner_results(
                    query=query,
                    video_path=video_path,
                    model_name=model_name,
                    results=tool_calls_result,
                    tools_so_far=self.limit_tools_so_far(tools_so_far, max_tools=int(os.environ.get("MAX_TOOLS_SO_FAR", "10"))),
                    visual_context=context_result.get("vlm_response", {}).get("video_context", ""),
                    timestamp_format=timestamp_format,
                    video_duration=video_duration,
                    return_ids=return_ids,
                    temperature=temperature,
                    **kwargs,
                )
                
                complete_results["iterative_reasoner"].append(iterative_reasoner_result)
                if return_ids:
                    completion_ids.extend(iterative_reasoner_result["completion_ids"])
                    prompt_ids.append((iterative_reasoner_result["prompt_ids"], iterative_reasoner_result["completion_ids"]))
                    attention_mask.append(iterative_reasoner_result["attention_mask"])
                    if not isinstance(iterative_reasoner_result["iterative_reasoner_results"], dict):
                        return iterative_reasoner_result["iterative_reasoner_results"], complete_results, completion_ids, prompt_ids, attention_mask

                final_answer = iterative_reasoner_result["iterative_reasoner_results"].get("final_answer", None)

                if not isinstance(iterative_reasoner_result["iterative_reasoner_results"].get("answerable", {}), dict):
                    iterative_reasoner_result["iterative_reasoner_results"]["answerable"] = {}

                if not bool(iterative_reasoner_result["iterative_reasoner_results"].get("answerable", {}).get("verdict", False)):
                    call_iterative_reasoner = True

                    if iterative_reasoner_result["iterative_reasoner_results"].get("recommended_tools", {}) is None:
                        iterative_reasoner_result["iterative_reasoner_results"]["recommended_tools"] = {}

                    if bool(iterative_reasoner_result["iterative_reasoner_results"].get("recommended_tools", {}).get("needed", False)):
                        tool_calls_result = self.get_tool_calls(iterative_reasoner_result["iterative_reasoner_results"], **kwargs)
                        tools_so_far = {
                            **tools_so_far,
                            **tool_calls_result.get("tool_calls", {}),
                        }
                        for tool_call in tool_calls_result.get("tool_calls", {}).values():
                            complete_results["args_validity"] = tool_call.get("args_validity", True)
                        if return_ids or "gemini" in model_name.lower():
                            if not complete_results["args_validity"]:
                                if "gemini" in model_name.lower():
                                    return final_answer, complete_results
                                return final_answer, complete_results, completion_ids, prompt_ids, attention_mask
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
                        complete_results["args_validity"] = False
                        tools_so_far = {**tools_so_far, **tool_calls_result.get("tool_calls", {})}
                        if return_ids or "gemini" in model_name.lower():
                            if not complete_results["args_validity"]:
                                if "gemini" in model_name.lower():
                                    return final_answer, complete_results
                                return final_answer, complete_results, completion_ids, prompt_ids, attention_mask
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
                    complete_results["args_validity"] = False
                    tools_so_far = {**tools_so_far, **tool_calls_result.get("tool_calls", {})}
                    if return_ids or "gemini" in model_name.lower():
                        if not complete_results["args_validity"]:
                            if "gemini" in model_name.lower():
                                return final_answer, complete_results
                            return final_answer, complete_results, completion_ids, prompt_ids, attention_mask
                else:
                    assert iterative_reasoner_result["iterative_reasoner_results"].get("final_answer", None) is not None
                    call_iterative_reasoner = False

        if return_ids:
            return final_answer, complete_results, completion_ids, prompt_ids, attention_mask
        return final_answer, complete_results
