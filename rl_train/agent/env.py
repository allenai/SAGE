import os
import numpy as np
import time
import threading
from typing import List, Dict, Any, Tuple
from sage.utils.utils import (
    SAGE_ITERATIVE_REASONER_PROMPT,
    SAGE_ITERATIVE_REASONER_MSG_PROMPT,
)
from sage.main import SAGE
from sage.src.functions.utils.utils import format_tool_descs
from sage.utils.json_parser import clean_json
from rl_train.agent.rewards import reward_funcs_registry, CODE_TO_NAME
from concurrent.futures import ThreadPoolExecutor, as_completed
from sage.src.functions.utils.utils import check_api_health

cores = os.cpu_count()

TOOL_CALL_MODEL = os.environ.get("TOOL_CALL_MODEL", "None")
ENV_WORKERS = int(os.environ.get("ENV_WORKERS", 1))
IS_MOLMO2 = os.environ.get("IS_MOLMO2", "False")
IS_MOLMO2 = IS_MOLMO2.lower() == "true"
VLLM_API_URL = os.environ.get("VLLM_API_URL", "None")
TRANSCRIBE_API_URL = os.environ.get("TRANSCRIBE_API_URL", "None")

def get_vllm_urls() -> List[str]:
    if VLLM_API_URL == "None" or not VLLM_API_URL:
        return VLLM_API_URL
    
    urls = [url.strip() for url in VLLM_API_URL.split(",") if url.strip()]
    if not urls:
        return VLLM_API_URL

    return urls

class SAGE_RL_Environment:
    """
    Main agent system that integrates the Context VLM with various video analysis tools for RL training.
    """

    def __init__(self, processor: object, tokenizer: object, config: object, timeout_seconds: int = 300):
        print("Initializing SAGE_RL_Environment...")
        self.tools_so_far = []
        self.processor = processor
        self.tokenizer = tokenizer
        self.config = config
        self.timeout_seconds = timeout_seconds
        if hasattr(self.config.reward_model, "short_traj_comp_coeff"):
            self.short_traj_comp_coeff = self.config.reward_model.short_traj_comp_coeff
        else:
            self.short_traj_comp_coeff = 1.0

        if hasattr(self.config.reward_model, "reward_fns"):
            self.reward_fns = self.config.reward_model.reward_fns.split("-")
            self.reward_fns = [CODE_TO_NAME[fn] for fn in self.reward_fns]
        else:
            self.reward_fns = ["accuracy", "format", "tool_call_and_answer_exclusive"]
        
        from openai import OpenAI
        self.tool_call_clients = [
            OpenAI(
                base_url=url, 
                api_key="not-needed"
            ) 
            for url in get_vllm_urls()
        ]

        self.generation_kwargs = {
                "n": 1,  # vLLM on each GPU generates only 1 in colocate mode
                "repetition_penalty": 1.0,
                "temperature": 0.0,
                "top_p": 0.9,
                "top_k": -1,
                "min_p": 0.0,
                "max_tokens": 4096,
                "guided_decoding": None,
            }

    def reset(self):
        self.tools_so_far = []
    
    def parse_llm_response(
        self,
        llm_response: str,
    ) -> Dict[str, Any]:

        try:
            while "<json>" in llm_response:
                llm_response = llm_response.split("<json>")[1].split("</json>")[0]
            llm_response, was_cleaned = clean_json(llm_response, return_was_cleaned=True)
            return llm_response, was_cleaned
        except Exception as e:
            # print("Error in parsing LLM response: ", e)
            return llm_response, False

    def process_llm_actions(self, llm_actions: List[str], is_placeholder_obs: List[bool]) -> Dict[str, Any]:
        processed_llm_actions = []
        for idx, llm_action in enumerate(llm_actions):
            res, was_cleaned = self.parse_llm_response(llm_action)
            if isinstance(res, dict) and not is_placeholder_obs[idx]:
                processed_llm_actions.append((res, True, was_cleaned))
            else:
                processed_llm_actions.append((res, False, was_cleaned))
        return processed_llm_actions
    
    def _process_single_llm_action(self, idx: int, llm_action: Tuple[Dict[str, Any], bool], 
                                   sage_agent: SAGE, is_cvlm: bool = False,
                                   num_actions: int = 16, video_path: str = None) -> Tuple[int, Dict[str, Any]]:
        """
        Process a single llm_action. This function is designed to be thread-safe.
        Returns a tuple of (index, result_dict) to maintain order.
        """
        llm_action, is_valid, was_cleaned = llm_action
        
        if is_valid and not isinstance(llm_action.get("recommended_tools", {}), dict):
            is_valid = False
        
        if is_valid and not isinstance(llm_action.get("recommended_tools", {}).get("tool_calls", []), list):
            is_valid = False
        
        if is_valid and not isinstance(llm_action.get("recommended_tools", {}).get("needed", False), bool):
            is_valid = False
        
        if is_valid and llm_action.get("recommended_tools", {}).get("needed", False) and len(llm_action.get("recommended_tools", {}).get("tool_calls", [])) > 0:
            try:
                tool_calls_result = self._call_with_timeout(
                    sage_agent.get_tool_calls,
                    llm_action,
                    timeout_seconds=self.timeout_seconds,
                )
                tool_call_name = list(tool_calls_result.get("tool_calls", {}).keys())[0].split("_#")[0]
                    
                for tool_call in tool_calls_result.get("tool_calls", {}).values():
                    args_validity = tool_call.get("args_validity", True)
                
                result = {
                    "result": tool_calls_result,
                    "state": "active",
                    "tool_calls": tool_calls_result.get("tool_calls", {}),
                    "args_validity": args_validity,
                    "tool_name": tool_call_name,
                    "was_cleaned": was_cleaned
                }
            except TimeoutError as e:
                print(f"Timeout getting tool calls: {str(e)}")
                result = {
                    "result": None,
                    "state": "invalid",
                    "tool_calls": {},
                    "args_validity": True,
                    "tool_name": None,
                    "was_cleaned": was_cleaned
                }

            if is_cvlm:
                result["video_context"] = llm_action.get("video_context", "")
        elif is_valid and llm_action.get("final_answer", None) is not None:
            result = {
                "result": llm_action.get("final_answer"),
                "state": "stop",
                "tool_calls": {},
                "args_validity": True,
                "tool_name": None,
                "was_cleaned": was_cleaned
            }
            if is_cvlm:
                result["video_context"] = llm_action.get("video_context", "")
        else:
            result = {
                "result": None,
                "state": "invalid",
                "tool_calls": {},
                "args_validity": True,
                "tool_name": None,
                "was_cleaned": was_cleaned
            }
            if is_cvlm:
                result["video_context"] = None
        
        if is_cvlm and (result["video_context"] is None or result["video_context"] == ""):
            result = {
                "result": None,
                "state": "invalid",
                "video_context": None,
                "tool_calls": {},
                "args_validity": True,
                "tool_name": None,
                "was_cleaned": was_cleaned
            }
        return result
    
    def _process_single_tool_result(self, idx: int, tool_calls_result: Dict[str, Any], 
                                   llm_action: Tuple[Dict[str, Any], bool], is_cvlm: bool,
                                   query: str, gt_answer: str, video_contexts: List[str],
                                   sage_agent: SAGE, video_path: str,
                                   video_duration_timestamp: str, turn: int, 
                                   max_turns: int, global_step: int) -> Tuple[int, Dict[str, Any]]:
        """
        Process a single tool call result. This function is designed to be thread-safe.
        Returns a tuple of (index, result_dict) to maintain order.
        """
        state = tool_calls_result["state"]
        args_validity = tool_calls_result["args_validity"]
        tool_name = tool_calls_result["tool_name"]
        was_cleaned = tool_calls_result["was_cleaned"]
        
        result = {
            "next_step_input": None,
            "reward": 0.0,
            "reward_dict": {},
            "done": False,
            "info": {},
            "video_context": None
        }
        
        if state == "invalid":
            if len(video_contexts) == 0:
                result["video_context"] = None
            result["next_step_input"] = None
            
            reward, reward_dict = self.compute_reward(llm_action, is_cvlm, True, query, gt_answer, args_validity, tool_name, self.tools_so_far[idx], turn, global_step, was_cleaned)
            
            result["reward"] = reward
            result["reward_dict"] = reward_dict
            result["done"] = True
            result["info"] = {"is_action_valid": 0, "final_answer": None, "tool_name": tool_name}
            
        elif state == "stop":
            if len(video_contexts) == 0:
                result["video_context"] = tool_calls_result["video_context"]
            result["next_step_input"] = None
            
            reward, reward_dict = self.compute_reward(llm_action, is_cvlm, True, query, gt_answer, args_validity, tool_name, self.tools_so_far[idx], turn, global_step, was_cleaned)
            if turn > 1:
                reward += (max_turns - turn) * self.short_traj_comp_coeff
            
            result["reward"] = reward
            result["reward_dict"] = reward_dict
            result["done"] = True
            result["info"] = {"is_action_valid": 1, "final_answer": tool_calls_result["result"], "tool_name": tool_name}
            
        else:
            assert len(self.tools_so_far[idx]) == turn, f"Tools so far length {len(self.tools_so_far[idx])} is not equal to turn {turn}"
            if len(video_contexts) == 0:
                result["video_context"] = tool_calls_result["video_context"]
            
            result["next_step_input"] = self.get_iterative_reasoner_inputs(
                query=query,
                video_path=video_path,
                prev_tool_call=tool_calls_result["result"]["tool_calls"],
                tools_so_far=sage_agent.limit_tools_so_far(self.tools_so_far[idx], max_tools=10),
                visual_context=result["video_context"],
                timestamp_format=video_duration_timestamp,
                sage_agent=sage_agent
            )
            
            reward, reward_dict = self.compute_reward(llm_action, is_cvlm, turn == max_turns, query, gt_answer, args_validity, tool_name, self.tools_so_far[idx], turn, global_step, was_cleaned)
            
            result["reward"] = reward
            result["reward_dict"] = reward_dict
            result["done"] = False
            result["info"] = {"is_action_valid": 1, "final_answer": "under_progress", "tool_name": tool_name}
        
        return (idx, result)
    
    
    def _call_with_timeout(self, func, *args, timeout_seconds: int = 300, **kwargs):
        """
        Thread-safe timeout wrapper that executes `func` in a dedicated thread
        and waits up to `timeout_seconds` for completion.
        """
        from concurrent.futures import ThreadPoolExecutor as _SingleExecutor
        from concurrent.futures import TimeoutError as _FuturesTimeout
        with _SingleExecutor(max_workers=1) as _executor:
            _future = _executor.submit(func, *args, **kwargs)
            try:
                return _future.result(timeout=timeout_seconds)
            except _FuturesTimeout:
                raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")

    def maybe_execute_tool_calls(
        self, 
        actor_rollout_wg: object, 
        llm_actions: List[Dict[str, Any]], 
        sage_agents: List[SAGE], 
        is_cvlm: bool = False,
        video_paths: List[str] = [],
    ) -> List[Dict[str, Any]]:
        """
        Execute tool calls using thread pooling for parallel processing.
        """
        num_actions = len(llm_actions)
        
        # Use ThreadPoolExecutor for parallel processing
        from tqdm import tqdm

        # Reduce thread pool size to avoid API rate limiting and connection exhaustion
        # max_workers = min(len(llm_actions), max(1, cores // 2), 8)  # Cap at 8 workers
        # max_workers = min(len(llm_actions), max(1, cores // 2))
        max_workers = ENV_WORKERS
        
        start_time = time.time()
        print(f"Processing tool calls...")
        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Prepare argument tuples and submit futures
                futures = []
                for idx, llm_action in enumerate(llm_actions):
                    future = executor.submit(
                        self._process_single_llm_action,
                        idx, llm_action, sage_agents[idx], is_cvlm, num_actions, video_paths[idx]
                    )
                    futures.append((idx, future))
                
                # Process completed futures with progress bar
                results = [None] * len(llm_actions)
                with tqdm(total=len(llm_actions), desc="Processing tool calls") as pbar:
                    # Create a mapping from future to index for O(1) lookup
                    future_to_idx = {f: idx for idx, f in futures}
                    for future in as_completed([f for _, f in futures]):
                        idx = future_to_idx[future]
                        results[idx] = future.result()
                        pbar.update(1)
        else:
            results = [self._process_single_llm_action(idx, llm_action, sage_agents[idx], is_cvlm, num_actions, video_paths[idx]) for idx, llm_action in tqdm(enumerate(llm_actions), total=len(llm_actions), desc="Processing tool calls")]

        # Log slow iterations
        elapsed = time.time() - start_time
        if len([r for r in results if r is not None]) > 0:
            avg_time = elapsed / len([r for r in results if r is not None])
            if avg_time > 1.0:  # If average time per task > 1 second
                print(f"Warning: Average processing time is {avg_time:.2f}s per task")
        
        # Process results and update tools_so_far
        tool_calls_results = []
        for idx, result in enumerate(results):
            tool_call_result = {
                "result": result["result"],
                "state": result["state"],
                "args_validity": result["args_validity"],
                "tool_name": result["tool_name"],
                "was_cleaned": result["was_cleaned"]
            }
            if is_cvlm:
                tool_call_result["video_context"] = result["video_context"]
            tool_calls_results.append(tool_call_result)
            
            # Update tools_so_far for active results
            if result["state"] == "active":
                self.tools_so_far[idx] = {**self.tools_so_far[idx], **result["tool_calls"]}
        
        return tool_calls_results

    
    def step(
        self, 
        actions: List[str], 
        actor_rollout_wg: object, 
        sage_agents: List[SAGE], 
        video_paths: List[str], 
        queries: List[str], 
        gt_answers: List[str],
        video_duration_timestamps: List[str],
        is_cvlm: bool = False,
        video_contexts: List[str] = [],
        turn: int = 0,
        max_turns: int = 10,
        is_placeholder_obs: List[bool] = [],
        global_step: int = 0,
        **kwargs) -> Tuple[str, List[bool], bool]:
        
        # Check if api servers are still running with retries
        if VLLM_API_URL != "None":
            max_retries = 5
            retry_delay = 0.5  # seconds
            urls = [url.strip() for url in VLLM_API_URL.split(",") if url.strip()]
            for url in urls:
                check_api_health(url, "VLLM API URL", max_retries=max_retries, base_delay=retry_delay)
        
        # Initialize tools_so_far if needed
        if len(self.tools_so_far) == 0:
            self.tools_so_far = [{} for _ in range(len(actions))]
        
        # Process LLM actions
        llm_actions = self.process_llm_actions(actions, is_placeholder_obs)
        
        # Execute tool calls
        tool_calls_results = self.maybe_execute_tool_calls(actor_rollout_wg, llm_actions, sage_agents, is_cvlm, video_paths)
        
        # Initialize result containers
        next_step_inputs = []
        rewards = []
        dones = []
        infos = []
        reward_infos = {}
        for reward_fn in self.reward_fns:
            reward_infos[reward_fn] = []
        if len(video_contexts) == 0:
            vid_contexts = []
        else:
            vid_contexts = video_contexts
        
        # Process each tool call result using thread pooling
        # Use ThreadPoolExecutor for parallel processing of tool results
        max_workers = ENV_WORKERS # Cap at 8 workers to avoid resource exhaustion
        
        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Prepare argument tuples and submit futures
                futures = []
                for idx, tool_calls_result in enumerate(tool_calls_results):
                    future = executor.submit(
                        self._process_single_tool_result,
                        idx, tool_calls_result, llm_actions[idx], is_cvlm,
                        queries[idx], gt_answers[idx], video_contexts,
                        sage_agents[idx], video_paths[idx],
                        video_duration_timestamps[idx], turn, max_turns, global_step
                    )
                    futures.append((idx, future))
                
                # Process completed futures and maintain order
                results = [None] * len(tool_calls_results)
                for idx, future in futures:
                    results[idx] = future.result()
        else:
            # Sequential processing if only 1 worker
            results = []
            for idx, tool_calls_result in enumerate(tool_calls_results):
                result = self._process_single_tool_result(
                    idx, tool_calls_result, llm_actions[idx], is_cvlm,
                    queries[idx], gt_answers[idx], video_contexts,
                    sage_agents[idx], video_paths[idx],
                    video_duration_timestamps[idx], turn, max_turns, global_step
                )
                results.append(result)
        
        # Process results and populate final arrays
        for idx, (_, result) in enumerate(results):
            next_step_inputs.append(result["next_step_input"])
            rewards.append(result["reward"])
            dones.append(result["done"])
            infos.append(result["info"])
            
            # Handle video contexts
            if len(video_contexts) == 0:
                vid_contexts.append(result["video_context"])
            
            # Aggregate reward information
            for k, v in result["reward_dict"].items():
                reward_infos[k].append(v)
        
        # Finalize results
        for reward_fn in self.reward_fns:
            reward_infos[reward_fn] = np.array(reward_infos[reward_fn])

        return next_step_inputs, np.array(rewards), np.array(dones), infos, vid_contexts, reward_infos

    def get_iterative_reasoner_inputs(
        self,
        query: str,
        video_path: str,
        prev_tool_call: Dict[str, Any] = None,
        tools_so_far: List[str] = [],
        visual_context: str = "None",
        timestamp_format: str = None,
        sage_agent: SAGE = None,
    ) -> Dict[str, Any]:

        video_info_prompt = f"Video Path: {video_path} of duration {timestamp_format}."
        prev_tool_call = sage_agent.format_tool_calls_args(prev_tool_call)
        query = (
            SAGE_ITERATIVE_REASONER_MSG_PROMPT.
                replace("<<<query>>>", query).
                replace("<<<video_info>>>", video_info_prompt).
                replace("<<<visual_context>>>", str(visual_context)).
                replace("<<<tools_so_far>>>", sage_agent.format_tool_calls(tools_so_far)).
                replace("<<<previous_tool_call>>>", prev_tool_call)
        )
        system_prompt = SAGE_ITERATIVE_REASONER_PROMPT.replace("<<<tools>>>", format_tool_descs(sage_agent.tools))
       
        if IS_MOLMO2:
            iterative_reasoner_input = [
                    {
                    "role": "user",
                    "content": system_prompt + "\n" + query,
                }
            ]
        else:
            iterative_reasoner_input = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": query,
                }
            ]
        
        return iterative_reasoner_input
    
    def format_tool_calls(self, tool_calls: Dict[str, Any]):
        tool_calls_str_args = ""
        for tool_call in tool_calls:
            tool_calls_str_args += f"{tool_call} with arguments {tool_calls[tool_call].get('arguments', {})}\n"
        return tool_calls_str_args
    
    def compute_reward(
        self, 
        llm_action: tuple, 
        is_cvlm: bool,
        is_final_step: bool, 
        question: str, 
        gt_answer: str,
        args_validity: bool,
        tool_name: str,
        tools_so_far: List[str] = [],
        turn: int = 0,
        global_step: int = 0,
        was_cleaned: bool = False,
    ) -> float:
        
        reward = 0.0
        reward_dict = {}
        for reward_fn in self.reward_fns:
            if not is_final_step and "accuracy" in reward_fn:
                value = 0.0
            else:
                value = reward_funcs_registry[reward_fn](
                    llm_action, is_cvlm=is_cvlm, question=question, 
                    gt_answer=gt_answer, args_validity=args_validity, 
                    tool_name=tool_name, tools_so_far=tools_so_far, 
                    turn=turn, global_step=global_step,
                    was_cleaned=was_cleaned
                )
            reward += value
            reward_dict[reward_fn] = value
        return reward, reward_dict