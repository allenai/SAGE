from typing import Dict, Any
from sage.src.api.gpt import GPT
import os
from typing import List
import json
import math

ACCURACY_PROMPT = """
Compare the model prediction and the ground truth and determine if they convey the same meaning for the question:

Question: {question}

Model Prediction: {hypothesis}
Ground Truth: {reference}

You MUST respond with the verdict as 'True' if they match semantically or 'False' if they don't match.
Answer in the following format:
Reasoning: <Reasoning for the verdict>
Verdict: <True/False>
"""

REASONING_TRACE_PROMPT = """Below is the reasoning trace for calling a sequence of tools for finding the answer to the question:

Question: {question}

Reasoning Trace: \n\n{reasoning_trace}

Predicted Answer: {predicted_answer}

You MUST respond with the verdict as 'True' if the reasoning trace makes sense for the question leading to the predicted answer or 'False' if it doesn't. 
You MUST penalize repetitive tool calls if they are not needed.
Answer in the following format:
Reasoning: <Reasoning for the verdict>
Verdict: <True/False>
"""

def format_reasoning_trace(tool_calls: Dict) -> str:
    """
    Format the reasoning trace of an entry.
    """
    def format_tool_call(tool_name: str, arguments: Dict, rationale: str, result: str, idx: int) -> str:
        """
        Format a tool call.
        """
        tool_name = tool_name.split("_#")[0]
        return f"Tool-Call-{idx}: \n\t - name: {tool_name}\n\t - arguments: {arguments}\n\t - rationale: {rationale}\n\t - result: {result}"

    tool_call_trace = ""
    count = 1
    
    if tool_calls and len(tool_calls) > 0:
        for tool_call in tool_calls:
            tool_name = tool_call
            arguments = tool_calls[tool_name].get("arguments")
            rationale = tool_calls[tool_name].get("rationale")
            result = tool_calls[tool_name].get("result")
            tool_call_trace += format_tool_call(tool_name, arguments, rationale, result, count)
            count += 1
            tool_call_trace += "\n"
    return tool_call_trace

# Lazy initialization of GPT for efficiency
gpt = GPT()

def llm_judge(prompt: str):
    # with torch.no_grad():
    response = gpt.get_response(prompt)[0]
    verdict = None
    if response:
        # Fallback: try to extract just the verdict
        if "Verdict:" in response:
            verdict = response.split("Verdict: ")[1].split("\n")[0].strip()
        is_correct = verdict is not None and "true" in verdict.strip().lower()
        if is_correct:
            return 1.0
    return 0.0

FORMAT_VALIDITY_PENALTY = float(os.environ.get("FORMAT_VALIDITY_PENALTY", "-0.1"))
def format_reward(llm_action: tuple, **kwargs):
    """Reward function that checks if the completion matches the required JSON format and fields."""
    is_cvlm = kwargs.get("is_cvlm", False)
    data, state, was_cleaned = llm_action

    if not state:
        return 0.0

    if is_cvlm:
        allowed_fields = {"recommended_tools", "final_answer", "video_context", "query_intent"}
    else:
        allowed_fields = {"recommended_tools", "final_answer", "answerable"}
    
    
    format_validity_penalty = FORMAT_VALIDITY_PENALTY
    format_validity_reward = -(FORMAT_VALIDITY_PENALTY / 2.0)
    
    keys = data.keys()
    extra_keys = set(keys) - allowed_fields
    missing_keys = allowed_fields - set(keys)
    if len(extra_keys) > 0 or len(missing_keys) > 0:
        return FORMAT_VALIDITY_PENALTY

    if "recommended_tools" in data and not isinstance(data["recommended_tools"], dict):
        return format_validity_penalty
    elif "recommended_tools" in data and isinstance(data["recommended_tools"], dict):
        allowed_tool_keys = {"needed", "why_no_tool", "tool_calls"}
        tool_keys = set(data["recommended_tools"].keys())
        extra_tool_keys = tool_keys - allowed_tool_keys
        missing_tool_keys = allowed_tool_keys - tool_keys
        if len(extra_tool_keys) > 0 or len(missing_tool_keys) > 0:
            return format_validity_penalty
    
    if not is_cvlm and "answerable" in data and not isinstance(data["answerable"], dict):
        return format_validity_penalty
    elif not is_cvlm and "answerable" in data and isinstance(data["answerable"], dict):
        allowed_answerable_keys = {"verdict", "reasoning"}
        answerable_keys = set(data["answerable"].keys())
        extra_answerable_keys = answerable_keys - allowed_answerable_keys
        missing_answerable_keys = allowed_answerable_keys - answerable_keys
        if len(extra_answerable_keys) > 0 or len(missing_answerable_keys) > 0:
            return format_validity_penalty

    return format_validity_reward


ARGS_VALIDITY_PENALTY = float(os.environ.get("ARGS_VALIDITY_PENALTY", "-0.1"))
def args_valid_penalty(llm_action: tuple, args_validity: bool, **kwargs):
    if not args_validity:
        return ARGS_VALIDITY_PENALTY
    else:
        return 0.0

REASONING_TRACE_REWARD = float(os.environ.get("REASONING_TRACE_REWARD", "0.1"))
def reasoning_trace_reward(
        llm_action: tuple, is_cvlm: bool, 
        tools_so_far: Dict[str, Any], question: str, turn: int, **kwargs
    ):
    if len(tools_so_far) < turn:
        return 0.0
    data, state, was_cleaned = llm_action
    if not isinstance(data, dict):
        return 0.0
    
    tool_call_history = ""
    for idx, (tool_name, tool_result) in enumerate(tools_so_far.items()):
        tool_name = tool_name.split("_#")[0]
        if (idx+1) == turn:
            tool_call_history += f"Step {idx + 1}: \n\t -> Tool Call: {tool_name} \n\t\t Arguments: {tool_result.get('arguments', {})} \n"
        else:
            tool_call_history += f"""Step {idx + 1}: \n\t -> Tool Call: {tool_name} \n\t\t Arguments: {tool_result.get('arguments', {})} \n\t\t Result: {tool_result.get('result', {})} \n"""

    prompt = f"""
    We are working on a video reasoning task where the model is given a question and a video and it needs to reason about the video and answer the question using a sequence of tools.

    We are currently at step {turn} of the reasoning process.

    Here is the tool call sequence for the reasoning process so far:
    {tool_call_history}

    Here is the question:
    {question}

    You MUST respond with the verdict as 'True' if the reasoning trace makes sense for the question and the tool call for the current turn is valid or 'False' if it doesn't.
    You MUST penalize repetitive tool calls if they are not needed.
    Answer in the following format:
    Reasoning: <Reasoning for the verdict>
    Verdict: <True/False>
    """

    response = gpt.get_response(prompt)[0]
    verdict = None
    if response:
        # Fallback: try to extract just the verdict
        if "Verdict:" in response:
            verdict = response.split("Verdict: ")[1].split("\n")[0].strip()
        is_correct = verdict is not None and "true" in verdict.strip().lower()
        if is_correct:
            return REASONING_TRACE_REWARD
        else:
            return -REASONING_TRACE_REWARD
    return 0.0


ARGS_REPEAT_PENALTY = float(os.environ.get("ARGS_REPEAT_PENALTY", "0.05"))
def args_repeat_penalty(
        llm_action: tuple, is_cvlm: bool, 
        tools_so_far: Dict[str, Any], question: str, turn: int, **kwargs
    ):
    tool_call_args_history = {}  # {tool_name: set of argument hashes}

    if len(tools_so_far) < turn:
        return 0.0
    
    for idx, (tool_name, tool_result) in enumerate(tools_so_far.items()):
        arguments = tool_result.get('arguments', {})
        tool_name = tool_name.split("_#")[0]
        # Create a hash of the arguments for comparison
        args_hash = hash(json.dumps(arguments, sort_keys=True))
        
        # Initialize set for this tool name if it doesn't exist
        if tool_name not in tool_call_args_history:
            tool_call_args_history[tool_name] = {}
        
        if args_hash not in tool_call_args_history[tool_name]:
            tool_call_args_history[tool_name][args_hash] = 0
        else:
            tool_call_args_history[tool_name][args_hash] += 1
        
        # Check if these exact arguments have been used before for this specific tool
        if (idx+1) == turn and tool_call_args_history[tool_name][args_hash] > 0:
            return -(ARGS_REPEAT_PENALTY * math.sqrt(tool_call_args_history[tool_name][args_hash]))
    
    return 0.0

JSON_CLEANED_PENALTY = float(os.environ.get("JSON_CLEANED_PENALTY", "-0.05"))
def json_cleaned_penalty(llm_action: tuple, was_cleaned: bool, **kwargs):
    if was_cleaned:
        return JSON_CLEANED_PENALTY
    else:
        return 0.0

def curr_accuracy_web_verb_penalty_reward(llm_action: tuple, tools_so_far: List[str], **kwargs):
    _model = kwargs.get("model", None)
    question = kwargs.get("question", None)
    gt_answer = kwargs.get("gt_answer", None)
    tools_history = [t.split("_#")[0] for t in tools_so_far.keys()]
    global_step = kwargs.get("global_step", 0)
    tool_counts = {}
    for t in tools_history:
        if t not in tool_counts:
            tool_counts[t] = 0
        tool_counts[t] += 1

    data, state, was_cleaned = llm_action
    if not isinstance(data, dict):
        return -2.0
    hypothesis = data.get("final_answer", None)
    is_cvlm = kwargs.get("is_cvlm", False)

    # reasoning_reward = 0.0
    if not is_cvlm:
        if "answerable" in data:
            if not isinstance(data["answerable"], dict):
                return -2.0
        elif not data.get("answerable", {}).get("verdict", False):
            return -2.0

    if hypothesis is None or not isinstance(hypothesis, str) or "could not produce an answer" in hypothesis.lower():
        return -0.5
    else:
        prompt = ACCURACY_PROMPT.format(question=question, hypothesis=hypothesis, reference=gt_answer)
        reward_llm = llm_judge(prompt)
        if reward_llm == 0.0 and not is_cvlm and len(tools_history) > 0:
            return -0.5
        elif (tool_counts.get("unified_web_search", 0) > 1 or tool_counts.get("verbal_transcript", 0) > 1) and reward_llm == 0.0:
            return -1.0
        elif (tool_counts.get("extract_parts_from_timestamp", 0) > 0 or tool_counts.get("identify_timestamps_visually", 0) > 0):
            reward_llm = 1.25 * reward_llm
        return reward_llm

reward_funcs_registry = {
    "format": format_reward,
    "args_valid": args_valid_penalty,
    "reasoning_trace": reasoning_trace_reward,
    "args_repeat": args_repeat_penalty,
    "json_cleaned_penalty": json_cleaned_penalty,
    "curr_accuracy_web_verb_penalty": curr_accuracy_web_verb_penalty_reward,
}

CODE_TO_NAME = {
    "format": "format",
    "args": "args_valid",
    "rt": "reasoning_trace",
    "args_re": "args_repeat",
    "json": "json_cleaned_penalty",
    "curr_web_verb_p_acc": "curr_accuracy_web_verb_penalty",
}