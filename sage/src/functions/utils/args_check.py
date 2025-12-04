import os
import re
from urllib.parse import urlparse
from typing import List

ENABLE_TIMESTAMP_DURATION_CHECK = os.environ.get("ENABLE_TIMESTAMP_DURATION_CHECK", "False").lower() == "true"

# Refactored: All functions now return (bool, error_message)
def check_file_exists(path: str, arg_name="file_path"):
    if not isinstance(path, str):
        return False, f"Argument '{arg_name}': Path must be a string, got {type(path).__name__}. Please provide a valid file path."

    if not os.path.isfile(path):
        return False, f"Argument '{arg_name}': File does not exist at path '{path}'. Please provide a valid file path."
    return True, ''


def check_valid_video_extension(path: str, arg_name="video_path"):
    if not isinstance(path, str):
        return False, f"Argument '{arg_name}': Path must be a string, got {type(path).__name__}. Please provide a valid file path."
    valid_exts = (".mp4", ".mov", ".avi", ".mkv", ".webm")
    if not path.lower().endswith(valid_exts):
        return False, f"Argument '{arg_name}': Invalid video file extension for '{path}'. Allowed extensions: {valid_exts}. Please provide a file with a valid video extension."
    return True, ''


def check_valid_extension(path: str, valid_exts=None, arg_name="file_path"):
    if not isinstance(path, str):
        return False, f"Argument '{arg_name}': Path must be a string, got {type(path).__name__}. Please provide a valid file path."
    if valid_exts is None:
        valid_exts = ('.mp4', '.mov', '.avi', '.mkv', '.webm')
    if not path.lower().endswith(valid_exts):
        return False, f"Argument '{arg_name}': Invalid file extension for '{path}'. Allowed extensions: {valid_exts}. Please provide a file with a valid extension."
    return True, ''


def check_valid_timestamp(ts: str, arg_name="timestamp"):
    # Ensure input is a non-empty string
    if not isinstance(ts, str) or not ts.strip():
        return False, f"Argument '{arg_name}': Timestamp must be a non-empty string. Got '{ts}'."
    # Accepts HH:MM:SS, MM:SS, or SS
    if not re.match(r"^(\d{1,2}:)?([0-5]?\d:)?[0-5]?\d$", ts):
        return False, f"Argument '{arg_name}': Invalid timestamp format '{ts}'. Expected format: 'HH:MM:SS', 'MM:SS', or 'SS'. Please provide a valid timestamp."
    return True, ''


def check_valid_url(url: str, arg_name="url"):
    if not isinstance(url, str):
        return False, f"Argument '{arg_name}': URL must be a string, got {type(url).__name__}. Please provide a valid URL."
    parsed = urlparse(url)
    if not (parsed.scheme and parsed.netloc):
        return False, f"Argument '{arg_name}': Invalid URL '{url}'. Please provide a valid URL with scheme (e.g., 'https') and domain."
    return True, ''


def check_in_allowed_values(value, allowed, arg_name="value"):
    if value not in allowed:
        return False, f"Argument '{arg_name}': Value '{value}' is not in allowed set {allowed}. Please choose one of the allowed values."
    return True, ''


def check_non_empty_string(s: str, arg_name="string"):
    if not isinstance(s, str) or not s.strip():
        return False, f"Argument '{arg_name}': String argument must be non-empty. Please provide a valid non-empty string."
    return True, ''


def check_all_files_exist(paths: List[str], arg_name="file_paths"):
    for i, path in enumerate(paths):
        if not isinstance(path, str):
            return False, f"Argument '{arg_name}': Item at index {i} is not a string (got {type(path).__name__}). Please provide a list of string file paths."
        if not os.path.isfile(path):
            return False, f"Argument '{arg_name}': File does not exist at path '{path}'. Please ensure all files exist."
    return True, ''


def parse_timestamp(ts: str, arg_name="timestamp"):
    """
    Parses a timestamp string (HH:MM:SS or MM:SS) into seconds.
    Returns (seconds, error_message) or (None, error_message) if invalid.
    """
    parts = ts.split(":")
    if len(parts) == 2:
        try:
            minutes = int(parts[0])
            seconds = int(parts[1])
            if not (0 <= minutes <= 59 and 0 <= seconds <= 59):
                return None, f"Argument '{arg_name}': Timestamp '{ts}' out of range. Minutes and seconds should be between 0 and 59."
            return minutes * 60 + seconds, ''
        except ValueError:
            return None, f"Argument '{arg_name}': Invalid timestamp format '{ts}'. Please use 'MM:SS'."
    elif len(parts) == 3:
        try:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2])
            if not (0 <= minutes <= 59 and 0 <= seconds <= 59):
                return None, f"Argument '{arg_name}': Timestamp '{ts}' out of range. Minutes and seconds should be between 0 and 59."
            return hours * 3600 + minutes * 60 + seconds, ''
        except ValueError:
            return None, f"Argument '{arg_name}': Invalid timestamp format '{ts}'. Please use 'HH:MM:SS'."
    else:
        return None, f"Argument '{arg_name}': Invalid timestamp format '{ts}'. Expected 'HH:MM:SS', 'MM:SS', or 'SS'."


def check_timestamp_range(start_ts: str, end_ts: str, start_arg="timestamp_start", end_arg="timestamp_end", enable_timestamp_duration_check=False):
    """
    Checks that both timestamps are valid, start <= end, and both are in valid range.
    Returns (bool, error_message).
    """
    valid_start, err_start = check_valid_timestamp(start_ts, arg_name=start_arg)
    if not valid_start:
        return False, f"Start timestamp error: {err_start}"
    valid_end, err_end = check_valid_timestamp(end_ts, arg_name=end_arg)
    if not valid_end:
        return False, f"End timestamp error: {err_end}"
    start_sec, err = parse_timestamp(start_ts, arg_name=start_arg)
    if start_sec is None:
        return False, f"Start timestamp error: {err}"
    end_sec, err = parse_timestamp(end_ts, arg_name=end_arg)
    if end_sec is None:
        return False, f"End timestamp error: {err}"
    if start_sec > end_sec:
        return False, f"Start timestamp ('{start_ts}') is after end timestamp ('{end_ts}'). Please ensure the start is before the end."
    elif start_sec == end_sec:
        return False, f"Start timestamp ('{start_ts}') and end timestamp ('{end_ts}') are the same. Please ensure the start is before the end."
    if enable_timestamp_duration_check:
        if end_sec - start_sec < 10:
            return False, f"The duration between start timestamp ('{start_ts}') and end timestamp ('{end_ts}') is less than 10 seconds. Please ensure the duration is at least 10 seconds."
        elif end_sec - start_sec > 1800:
            return False, f"The duration between start timestamp ('{start_ts}') and end timestamp ('{end_ts}') is greater than 1800 seconds. Please ensure the duration is at most 1800 seconds."
    return True, ''


class ArgsValidator:
    """
    Validates presence and value of required arguments using provided check functions.
    Usage:
        validator = ArgsValidator({
            'video_path': (True, check_file_exists),
            'timestamp_start': (True, check_valid_timestamp),
            'timestamp_end': (True, check_valid_timestamp),
            'url': (False, check_valid_url),
        })
        ok, errors = validator.validate(args_dict)
    """
    def __init__(self, arg_checks):
        """
        arg_checks: dict of arg_name -> (required: bool, check_fn: callable)
        """
        self.arg_checks = arg_checks

    def validate(self, args: dict):
        errors = {}
        # Check for missing required arguments
        for arg, (required, check_fn) in self.arg_checks.items():
            if required and arg not in args:
                errors[arg] = f"Missing required argument '{arg}'. Please provide this argument."
                continue
            if arg in args:
                valid, err = check_fn(args[arg])
                if not valid:
                    errors[arg] = err
        return (len(errors) == 0), errors

def _at_least_one_present(args, keys):
    present = [k for k in keys if k in args and args[k] is not None and str(args[k]).strip()]
    if not present:
        return False, f"At least one of {keys} must be provided and non-empty. Please provide at least one of these arguments."
    return True, ''

def chain_checks(*check_fns):
    """Chain multiple check functions, returning the first error encountered."""
    def chained(value):
        for fn in check_fns:
            valid, err = fn(value)
            if not valid:
                return valid, err
        return True, ''
    return chained

# Tool argument validators
verbal_transcript_args_validator = ArgsValidator({
    'video_path': (True, chain_checks(lambda v: check_file_exists(v, arg_name='video_path'), lambda v: check_valid_extension(v, arg_name='video_path'))),
})

def patch_validator_with_timestamp_range(validator, start_key='timestamp_start', end_key='timestamp_end', enable_timestamp_duration_check=False):
    old_validate = validator.validate
    def new_validate(args):
        ok, errors = old_validate(args)
        if start_key in args and end_key in args:
            ok_range, err_range = check_timestamp_range(args[start_key], args[end_key], start_arg=start_key, end_arg=end_key, enable_timestamp_duration_check=enable_timestamp_duration_check)
            if not ok_range:
                errors['timestamp_range'] = err_range
                ok = False
        elif start_key in args or end_key in args:
            errors['timestamp_range'] = f"Both '{start_key}' and '{end_key}' must be provided together."
            ok = False
        return ok, errors
    validator.validate = new_validate

perform_reasoning_args_validator = ArgsValidator({
    'query': (True, lambda v: check_non_empty_string(v, arg_name='query')),
    'media_paths': (False, lambda v: (True, '') if v is None else (
        (isinstance(v, list) and all(isinstance(item, str) for item in v) and check_all_files_exist(v, arg_name='media_paths')[0],
         'media_paths must be a list of existing files. Please provide a list of valid file paths.') if isinstance(v, list) and all(isinstance(item, str) for item in v) else
        (False, 'media_paths must be a list of strings. Please provide a list of valid file paths.'))),
})

unified_web_search_args_validator = ArgsValidator({
    'query': (False, lambda v: check_non_empty_string(v, arg_name='query')),
    'image_path': (False, lambda v: check_non_empty_string(v, arg_name='image_path')),
    'search_type': (False, lambda v: check_in_allowed_values(v, ['web', 'image'], arg_name='search_type')),
    'num_results': (False, lambda v: (isinstance(v, int) and 1 <= v <= 10, 'num_results must be int between 1 and 10. Please provide an integer between 1 and 10.' ) if v is not None else (True, '')),
})

parse_web_data_args_validator = ArgsValidator({
    'website_url': (True, lambda v: check_valid_url(v, arg_name='website_url')),
})

# Patch ArgsValidator to support at-least-one-required logic for unified_web_search
old_validate = unified_web_search_args_validator.validate

def new_validate(args):
    ok, errors = old_validate(args)
    ok2, err2 = _at_least_one_present(args, ['query', 'image_path'])
    if not ok2:
        errors['query|image_path'] = err2
        ok = False
    return ok, errors

unified_web_search_args_validator.validate = new_validate

extract_parts_from_timestamp_args_validator = ArgsValidator({
    'video_path': (True, chain_checks(lambda v: check_file_exists(v, arg_name='video_path'), lambda v: check_valid_extension(v, arg_name='video_path'))),
    'extract_type': (False, lambda v: check_in_allowed_values(v, ['frames', 'subclips'], arg_name='extract_type')),
})


identify_timestamps_visually_args_validator = ArgsValidator({
    'video_path': (True, chain_checks(lambda v: check_file_exists(v, arg_name='video_path'), lambda v: check_valid_extension(v, arg_name='video_path'))),
    'event': (True, lambda v: check_non_empty_string(v, arg_name='event')),
})

# Patch all relevant validators
patch_validator_with_timestamp_range(verbal_transcript_args_validator, enable_timestamp_duration_check=ENABLE_TIMESTAMP_DURATION_CHECK)
patch_validator_with_timestamp_range(extract_parts_from_timestamp_args_validator)
patch_validator_with_timestamp_range(identify_timestamps_visually_args_validator, enable_timestamp_duration_check=ENABLE_TIMESTAMP_DURATION_CHECK)