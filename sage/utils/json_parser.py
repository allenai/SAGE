import json
import re

class JSONParsingError(Exception):
    """Custom exception for JSON parsing failures."""
    def __init__(self, message, invalid_json_string=None):
        self.message = message
        self.invalid_json_string = invalid_json_string
        full_message = f"{message}"
        # if invalid_json_string:
            # full_message += f"\n--- INVALID JSON STRING TRIED ---\n{invalid_json_string}"
        super().__init__(full_message)

def _escape_newlines_in_strings(json_str: str) -> str:
    """
    Escapes unescaped newline characters (\n) only when they are inside a string literal.
    This is a stateful parser to avoid breaking the overall JSON structure.
    """
    repaired_chars = []
    in_string = False
    is_escaped = False

    for char in json_str:
        if is_escaped:
            # The previous char was '\', so this char is literal.
            repaired_chars.append(char)
            is_escaped = False
            continue

        if char == '\\':
            is_escaped = True
            repaired_chars.append(char)
            continue

        if char == '"':
            in_string = not in_string

        # The primary fix: if we are inside a string and find a newline, escape it.
        if in_string and char == '\n':
            repaired_chars.append('\\n')
        else:
            repaired_chars.append(char)

    return "".join(repaired_chars)

def _remove_json_comments(json_str: str) -> str:
    """
    Removes comments (starting with # or //) from a JSON string, but only outside of string literals.
    """
    result = []
    in_string = False
    is_escaped = False
    i = 0
    while i < len(json_str):
        char = json_str[i]
        if is_escaped:
            result.append(char)
            is_escaped = False
        elif char == '\\':
            result.append(char)
            is_escaped = True
        elif char == '"':
            result.append(char)
            in_string = not in_string
        elif not in_string and char == '#':
            # Skip until end of line
            while i < len(json_str) and json_str[i] != '\n':
                i += 1
            continue
        elif not in_string and json_str[i:i+2] == '//':
            # Skip until end of line
            i += 2
            while i < len(json_str) and json_str[i] != '\n':
                i += 1
            continue
        else:
            result.append(char)
        i += 1
    return ''.join(result)

def _close_unclosed_structures(json_str: str) -> str:
    """
    Closes any unclosed braces { } and brackets [ ] at the end of the JSON string.
    This handles cases where the JSON is truncated.
    """
    in_string = False
    is_escaped = False
    open_braces = 0
    open_brackets = 0
    
    for char in json_str:
        if is_escaped:
            is_escaped = False
            continue
        
        if char == '\\':
            is_escaped = True
            continue
        
        if char == '"':
            in_string = not in_string
            continue
        
        if not in_string:
            if char == '{':
                open_braces += 1
            elif char == '}':
                open_braces -= 1
            elif char == '[':
                open_brackets += 1
            elif char == ']':
                open_brackets -= 1
    
    # Close unclosed structures in reverse order (brackets first, then braces)
    closing = ''
    closing += ']' * open_brackets
    closing += '}' * open_braces
    
    return json_str + closing

def _repair_json_string(json_str: str) -> str:
    """
    Applies a chain of repairs to a string to fix common JSON syntax errors.
    The repairs are ordered from safest and most fundamental to more speculative.
    """
    repaired_str = json_str

    # --- Repair 1: Escape unescaped newlines within string literals. ---
    repaired_str = _escape_newlines_in_strings(repaired_str)

    # --- NEW REPAIR RULE: Remove comments (lines starting with # or //) outside of string literals ---
    repaired_str = _remove_json_comments(repaired_str)
    
    # --- NEW REPAIR: Close unclosed braces and brackets ---
    repaired_str = _close_unclosed_structures(repaired_str)

    # --- IMPROVED REPAIR: Add missing commas between elements (move earlier, improve regex) ---
    # Insert a comma after a value (string, number, boolean, null, object, array) if followed by a quoted key
    repaired_str = re.sub(
        r'((?:"[^"]*"|\d+|true|false|null|\}|\]))\s*(")',
        lambda m: m.group(1).rstrip() + ',\n' + m.group(2),
        repaired_str
    )

    # --- NEW REPAIR RULE: Fix malformed object structures with unexpected commas and newlines ---
    repaired_str = re.sub(r',\s*\n\s*"', r', "', repaired_str)

    # --- NEW REPAIR RULE: Remove keys that have no value. ---
    repaired_str = re.sub(r',\s*("[^"]+")\s*(?=[,\}])', '', repaired_str)

    # --- NEW REPAIR RULE: Remove any key (with or without a comma before) not followed by a colon. ---
    # This handles cases where a key is present but missing a colon, e.g., { "foo", "bar": 1 } or { "foo" }
    repaired_str = re.sub(r'(,?\s*"[^"]+"\s*)(?=[,\}])', '', repaired_str)

    # --- IMPROVED REPAIR: Insert missing colon between key and value, but only if not already present ---
    # Handles cases like { "foo" "bar" } -> { "foo": "bar" }
    repaired_str = re.sub(
        r'("[^"]+")\s+(?!:)("[^"]+"|\d+|true|false|null|\[|\{)',
        r'\1: \2',
        repaired_str
    )

    # --- NEW REPAIR: Remove keys at the end of an object not followed by a colon and value ---
    # Handles: { ..., "key"} or { ..., "key" ]
    repaired_str = re.sub(r'(,?\s*"[^"]+")\s*([}\]])', r'\2', repaired_str)

    # --- NEW REPAIR: Clean up string values with trailing commas and spaces before a closing brace/bracket ---
    # Handles: "value, "} -> "value"}
    repaired_str = re.sub(r'("[^"]*?),\s*([}\]])', r'\1\2', repaired_str)

    # --- NEW REPAIR: Ensure any key not followed by a value gets null ---
    # Handles: "key": } or "key": ,
    repaired_str = re.sub(r'("[^"]+")\s*:\s*([}\],])', r'\1: null\2', repaired_str)

    # --- NEW REPAIR: Replace any key at the end of an object (before }) that is not followed by a colon and value with null ---
    # Handles: { ..., "key"} -> { ..., "key": null}
    repaired_str = re.sub(r'("[^"]+")\s*}', r'\1: null}', repaired_str)

    # --- Repair 2: Fix missing closing brace `}` for `arguments` objects. ---
    repaired_str = re.sub(
        r'(:\s*"[^"]*?")(\s*[^,}\]]+?)\s*(,\s*"rationale":)',
        r'\1}\3',
        repaired_str,
        flags=re.DOTALL
    )

    # --- Repair X: Replace missing array values with empty array ---
    repaired_str = re.sub(r'("[^"]+")\s*:\s*\]', r'\1: []', repaired_str)

    # --- Repair Y: Fill other missing values with null ---
    repaired_str = re.sub(r'("[^"]+")\s*:\s*(?=[,\}\]])', r'\1: null', repaired_str)
    
    # --- Repair 3: Remove illegal trailing commas. ---
    repaired_str = re.sub(r',\s*([\]\}])', r'\1', repaired_str)

    # --- Repair 5: Remove hallucinated text at the end of an object. ---
    repaired_str = re.sub(
        r'("\w+"\s*:\s*(?:".*?"|\d+|true|false|null))\s*[^"\},]+?\s*(\})',
        r'\1\2',
        repaired_str,
        flags=re.DOTALL
    )

    # --- Repair 6: Ensure the `final_answer` key exists for schema compliance. ---
    if '"final_answer"' not in repaired_str:
        if repaired_str.strip().endswith('}'):
            last_brace_index = repaired_str.rfind('}')
            content_before_brace = repaired_str[:last_brace_index].strip()
            comma = ',' if content_before_brace and not content_before_brace.endswith(',') and not content_before_brace.endswith('{') else ''
            repaired_str = repaired_str[:last_brace_index] + f'{comma}\n  "final_answer": null\n' + repaired_str[last_brace_index:]

    return repaired_str

def clean_json(malformed_json_string: str, return_was_cleaned: bool = False) -> dict:
    """
    Repairs and parses a JSON string that is expected to be a single object.
    It attempts a direct parse first, and if that fails, applies a series of repairs.
    """
    # Attempt a direct parse first for valid JSON.
    try:
        result = json.loads(malformed_json_string)
        if return_was_cleaned:
            return result, False
        return result
    except json.JSONDecodeError:
        # If it fails, proceed to the repair logic.
        pass
    was_cleaned = True

    repaired_string = _repair_json_string(malformed_json_string)
    try:
        if return_was_cleaned:
            return json.loads(repaired_string), True
        return json.loads(repaired_string)
    except json.JSONDecodeError as e:
        # If it still fails, raise the custom error with the repaired string for debugging.
        raise JSONParsingError(
            message=f"Failed to decode JSON after all repair attempts. Error: {e}",
            invalid_json_string=repaired_string
        ) from e

# --- TESTING ---
if __name__ == "__main__":
    # Test case from the "Invalid control character" error
    test_case_newline = '''
{
  "extra_tool_to_implement": {
    "name": "extract_on_screen_text",
    "output_format": "[{'timestamp': 'HH:MM:SS', 'text': 'string'}],
"
  }
}'''

    all_test_cases = {
        "Unescaped Newline in String": test_case_newline,
        "Trailing Comma": '''{ "key": "value", }''',
        "Missing Comma": '''{ "key1": "value1" "key2": "value2" }''',
        "Hallucination at End of Object": '''{ "key": "value" some junk text }''',
        "Already Valid JSON": '''{ "key": "value", "final_answer": null }''',
        # New test cases for missing values
        "Missing Value - Comma": '''{ "foo": "bar", "missing": , "baz": 1 }''',
        "Missing Value - Brace": '''{ "foo": "bar", "missing": }''',
        "Missing Value - Bracket": '''{ "foo": "bar", "missing": ] }''',
        # Test case for # comment
        "Comment with #": '''{
  "name": null, "timestamps": {
    "start": null, #H:MM:SS
    "end": "00:09:06"},
  "final_answer": null
}''',
        # Test case for // comment
        "Comment with //": '''{
  "foo": 1, // this is a comment\n  "bar": 2
}''',
    }

    for name, test_str in all_test_cases.items():
        print(f"--- TESTING: {name} ---")
        try:
            result = clean_json(test_str)
            print("Successfully parsed!")
            print(json.dumps(result, indent=2))
        except JSONParsingError as e:
            print(f"Failed to parse:\n{e}")
        print("\n" + "="*40 + "\n")