import os
import ast
import json
from datetime import datetime
from dateutil import parser
from dateutil.relativedelta import relativedelta
import importlib
import sys
from google.cloud import storage
import re
import requests
import time

# --- Tool Dispatcher with Argument Validation ---
from sage.src.functions.utils.args_check import (
    verbal_transcript_args_validator,
    perform_reasoning_args_validator,
    unified_web_search_args_validator,
    extract_parts_from_timestamp_args_validator,
    identify_timestamps_visually_args_validator,
    parse_web_data_args_validator,
)

def check_api_health(api_url, service_name, max_retries=3, base_delay=1):
    """Common function to check API health with retry logic"""
    if api_url == "None":
        return
    if "/v1" in api_url:
        health_url = api_url.replace("/v1", "/health")
    else:
        health_url = api_url + "/health"
    
    for attempt in range(max_retries):
        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                return

            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"Retrying in {delay} seconds...")            
                time.sleep(delay)
        except Exception as e:
            # print(f"Error checking {service_name} health: {e}")
            time.sleep(base_delay)
        
    raise Exception(f"{service_name} is not healthy at {health_url}")

validator_mapping = {
    "verbal_transcript": verbal_transcript_args_validator,
    "perform_reasoning": perform_reasoning_args_validator,
    "unified_web_search": unified_web_search_args_validator,
    "extract_parts_from_timestamp": extract_parts_from_timestamp_args_validator,
    "identify_timestamps_visually": identify_timestamps_visually_args_validator,
    "parse_web_data": parse_web_data_args_validator,
}

def is_url(path: str) -> bool:
    """Check if the given path is a URL."""
    return re.match(r"^https?://", path) is not None


def upload_to_gcp_bucket(local_path: str, bucket_name: str, dest_blob_name: str) -> str:
    """Uploads a file to GCP bucket and returns the public URL."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(dest_blob_name)
    blob.upload_from_filename(local_path)
    # Make the blob publicly accessible
    blob.make_public()
    return blob.public_url

def format_tool_descs(tools: list) -> str:
    tool_descs = [{'type': 'function', 'function': f} for f in tools]
    return "<tools>\n" + '\n'.join([json.dumps(f, ensure_ascii=False) for f in tool_descs]) + "\n</tools>"

def get_relative_time(timestamp):
    tweet_datetime = parser.parse(timestamp)
    now = datetime.now(tweet_datetime.tzinfo)
    delta = relativedelta(now, tweet_datetime)

    def format_delta(delta):
        if delta.years > 0:
            return f"{delta.years} years ago"
        elif delta.months > 0:
            return f"{delta.months} months ago"
        elif delta.days > 0:
            return f"{delta.days} days ago"
        elif delta.hours > 0:
            return f"{delta.hours} hours ago"
        elif delta.minutes > 0:
            return f"{delta.minutes} minutes ago"
        else:
            return "just now"

    tweet_timestamp = format_delta(delta)
    return tweet_timestamp


def parse_json_file(file_path):
    """
    Parses a JSON file and returns its content.

    :param file_path: Path to the JSON file to be parsed.
    :return: Content of the JSON file
    """
    with open(file_path, "r") as file:
        content = json.load(file)
    return content


def parse_txt_file(file_path):
    """
    Parses a text file and returns its content as a string.

    :param file_path: Path to the text file to be parsed.
    :return: Content of the text file as a string.
    """
    with open(file_path, "r") as file:
        content = file.read()
    return content


def _get_function_info(node):
    """
    Extracts information from a function node.

    :param node: AST node representing a function.
    :return: Dictionary with function name, arguments, return type, and description.
    """
    docstring = ast.get_docstring(node) or ""
    # Split docstring into main description and args section
    parts = docstring.split("Args:")
    main_description = parts[0].strip()
    args_section = parts[1].strip() if len(parts) > 1 else ""

    # Parse argument descriptions from the args section
    arg_descriptions = {}
    if args_section:
        for line in args_section.split("\n"):
            line = line.strip()
            if ":" in line and not line.startswith("Returns:"):
                arg_name, desc = line.split(":", 1)
                arg_name = arg_name.strip()
                desc = desc.strip()
                if arg_name and desc:
                    arg_descriptions[arg_name] = desc

    # Map Python types to JSON Schema types
    type_mapping = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "list": "array",
        "dict": "object",
        "None": "null",
    }

    func_info = {
        "name": node.name,
        "description": main_description,
        "parameters": {"type": "object", "properties": {}, "required": []},
    }

    # Get all non-self arguments
    args = [arg for arg in node.args.args if arg.arg != "self"]
    # Get default values
    defaults = node.args.defaults

    # Calculate the number of required arguments (those without defaults)
    num_required = len(args) - len(defaults)

    # Process all arguments
    for i, arg in enumerate(args):
        arg_name = arg.arg
        if arg.annotation:
            # Handle different types of annotations
            if isinstance(arg.annotation, ast.Name):
                python_type = arg.annotation.id
                arg_type = type_mapping.get(python_type, "any")
            elif isinstance(arg.annotation, ast.Subscript):
                # Handle subscripted types like List[str]
                if isinstance(arg.annotation.value, ast.Name):
                    base_type = arg.annotation.value.id
                    if isinstance(arg.annotation.slice, ast.Index):
                        if isinstance(arg.annotation.slice.value, ast.Name):
                            inner_type = type_mapping.get(arg.annotation.slice.value.id, "any")
                            if base_type in ["List", "list"]:
                                arg_type = {
                                    "type": "array",
                                    "items": {"type": inner_type},
                                }
                            elif base_type in ["Dict", "dict"]:
                                arg_type = {
                                    "type": "object",
                                    "additionalProperties": {"type": inner_type},
                                }
                            else:
                                arg_type = type_mapping.get(base_type, "any")
                        else:
                            arg_type = type_mapping.get(base_type, "any")
                    else:
                        arg_type = type_mapping.get(base_type, "any")
                else:
                    arg_type = "any"
            else:
                arg_type = "any"
        else:
            arg_type = "string"  # Default type if not specified

        arg_description = arg_descriptions.get(arg_name, "")
        func_info["parameters"]["properties"][arg_name] = {
            "type": arg_type,
            "description": arg_description,
        }

        # Add to required list if this argument doesn't have a default
        if i < num_required:
            func_info["parameters"]["required"].append(arg_name)
        else:
            # This argument has a default value
            default_idx = i - num_required
            if isinstance(defaults[default_idx], ast.Constant):
                default_value = defaults[default_idx].value
            else:
                default_value = "unknown"
            func_info["parameters"]["properties"][arg_name]["default"] = default_value

    return func_info


def _get_functions_from_file(file_path):
    """
    Parses a Python file and extracts information about all functions defined in it.

    :param file_path: Path to the Python file to be parsed.
    :return: List of dictionaries with information about each function.
    """
    with open(file_path, "r") as file:
        tree = ast.parse(file.read(), filename=file_path)

    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    return [_get_function_info(func) for func in functions]


def get_functions_from_folder(folder_path, drop_tool_call_files: list = []):
    """
    Parses all Python files in a folder and extracts information about all functions defined in them.

    :param folder_path: Path to the folder containing Python files.
    :return: Dictionary with file names as keys and lists of function information as values.
    """
    functions = []
    function_dispatcher = {}
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".py") and file not in drop_tool_call_files:
                file_path = os.path.join(root, file)
                function_module = _load_module_from_folder(root, file.split(".")[0])
                function_dispatcher = {
                    **function_dispatcher,
                    **_create_dispatcher(function_module),
                }
                functions.extend(_get_functions_from_file(file_path))
    # print(f"Tools available: {list(function_dispatcher.keys())}, drop_tool_call_files: {drop_tool_call_files}, tool_call_files: {tool_call_files}")
    return functions, function_dispatcher


def _create_dispatcher(module):
    """
    Creates a function dispatcher based on the given module, including methods with 'self' as their first argument.
    Attaches argument validators if available.
    """
    dispatcher = {}
    module_ast = ast.parse(open(module.__file__).read())

    for node in module_ast.body:
        if isinstance(node, ast.FunctionDef):
            func_info = _get_function_info(node)
            validator = validator_mapping.get(func_info["name"])
            # Check if the function has 'self' as the first argument
            if node.args.args and node.args.args[0].arg == "self":
                # If class name is provided, associate method with the class
                func_info = _get_function_info(node)
                dispatcher[func_info["name"]] = {
                    "is_method": True,
                    "function": getattr(module, func_info["name"]),
                    "args": list(func_info["parameters"]["properties"].keys()),
                    "args_validator": validator,
                }
            else:
                # Regular function without 'self'
                func_info = _get_function_info(node)
                dispatcher[func_info["name"]] = {
                    "is_method": False,
                    "function": getattr(module, func_info["name"]),
                    "args": list(func_info["parameters"]["properties"].keys()),
                    "args_validator": validator,
                }
    return dispatcher


def _load_module_from_folder(folder_path, module_name):
    """
    Dynamically loads a Python module from a folder.

    :param folder_path: The path to the folder containing the module.
    :param module_name: The name of the module to load (e.g., 'my_module').
    :return: The loaded module.
    """
    # Build the path to the module
    module_path = os.path.join(folder_path, f"{module_name}.py")

    # Ensure the module file exists
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Module {module_name} not found in {folder_path}")

    # Load the module
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)

    # Add the module to sys.modules
    sys.modules[module_name] = module

    # Execute the module
    spec.loader.exec_module(module)

    return module
