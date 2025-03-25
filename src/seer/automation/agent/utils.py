import json
import re
import time
import random
import logging
from functools import wraps
from typing import Any, Callable, TypeVar, cast

import tree_sitter_languages

T = TypeVar("T")


def with_exponential_backoff(
    max_retries: int = 5,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    error_types: tuple = ("overloaded_error",),
):
    """
    Decorator that implements retry with exponential backoff for API calls.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds between retries
        max_delay: Maximum delay in seconds between retries
        backoff_factor: Factor to increase the delay with each retry
        jitter: Whether to add randomness to the delay to prevent synchronized retries
        error_types: Tuple of error types to retry on
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            retries = 0
            delay = initial_delay
            
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_info = str(e)
                    retry_error = False
                    
                    # Check if this is an error we should retry
                    for error_type in error_types:
                        if f"'type': '{error_type}'" in error_info:
                            retry_error = True
                            break
                    
                    # If not a retryable error, or we've used all retries, raise the exception
                    if not retry_error or retries >= max_retries:
                        logging.error(f"Failed after {retries} retries: {error_info}")
                        raise
                    
                    retries += 1
                    # Calculate delay with exponential backoff
                    actual_delay = min(delay * (backoff_factor ** (retries - 1)), max_delay)
                    # Add jitter if enabled (up to 20% random variation)
                    if jitter:
                        actual_delay = actual_delay * (0.8 + 0.4 * random.random())
                    
                    logging.warning(
                        f"Retrying request after error: {error_info}. "
                        f"Retry {retries}/{max_retries}, waiting {actual_delay:.2f}s"
                    )
                    time.sleep(actual_delay)
            
        return wrapper
    return decorator

def replace_newlines_not_in_quotes(value):
    # Split the string by both single and double quotes, replace \n outside quotes, and reassemble
    parts = re.split(r'(["\'][^"\']*["\'])', value)
    for i, part in enumerate(parts):
        if i % 2 == 0:  # Not inside quotes
            part = part.replace("\\n", "\n")
        parts[i] = part
    return "".join(parts)


def parse_json_with_keys(json_str: str, valid_keys: list[str]) -> dict[str, Any]:
    """
    NOTE: This function will only work if the json is not nested, and also doesn't work on arrays.

    Parses a JSON string and extracts key-value pairs where the key is in the list of valid keys.

    This function uses the tree-sitter library to parse the JSON string into a syntax tree,
    then iterates over the tree to find key-value pairs ('pair' nodes). If a key is not in the
    list of valid keys, it checks for a previous sibling node that is a valid key and associates
    the value with that key instead. The result is a dictionary containing the valid key-value pairs.

    Args:
        json_str: A string containing the JSON data.
        valid_keys: A list of strings representing the keys to extract values for.

    Returns:
        A dictionary where each key is from the list of valid keys and the value is the associated
        value from the JSON string.
    """
    json_parser = tree_sitter_languages.get_parser("json")
    tree = json_parser.parse(bytes(json_str, "utf-8"))

    parsed_json = []

    for node in tree.root_node.children[0].children:
        if node.type == "pair":
            key = node.children[0].text[1:-1].decode("utf-8")
            if key not in valid_keys:
                key_contains_invalid_chars = any(
                    char in key for char in [" ", "\n", "\t", "\r", "\f", "\b", '"', "'", ":", ","]
                )
                prev_sibling = node.prev_named_sibling
                while prev_sibling and (
                    prev_sibling.type != "pair"
                    or prev_sibling.children[0].text[1:-1].decode("utf-8") not in valid_keys
                ):
                    prev_sibling = prev_sibling.prev_named_sibling

                if key_contains_invalid_chars and prev_sibling and prev_sibling.type == "pair":
                    correct_key = prev_sibling.children[0].text[1:-1].decode("utf-8")
                    is_previous_child_str = prev_sibling.children[2].type == "string"
                    start_byte = (
                        prev_sibling.children[2].start_byte + 1
                        if is_previous_child_str
                        else prev_sibling.children[2].start_byte
                    )
                    is_current_child_str = node.children[2].type == "string"
                    end_byte = (
                        node.children[2].end_byte - 1
                        if is_current_child_str
                        else node.children[2].end_byte
                    )
                    # It's fine that we let invalid parses just become strings...
                    parsed_json.append(
                        (correct_key, tree.root_node.text[start_byte:end_byte].decode("utf-8"))
                    )
            else:
                child_node = node.children[2]
                value = (
                    child_node.text[1:-1].decode("utf-8")
                    if child_node.type == "string"
                    else child_node.text.decode("utf-8")
                )

                if child_node.type != "string":
                    json_str = f'{{ "value": {value} }}'
                    value = json.loads(json_str)["value"]

                parsed_json.append((key, value))

    json_dict = {}
    for key, value in parsed_json:
        if isinstance(value, str):
            value = replace_newlines_not_in_quotes(value)
        json_dict[key] = value

    return json_dict
