import json
import re
from typing import Any

import tree_sitter_languages


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


def extract_json_from_text(string: str | None) -> dict | None:
    """
    If a string has text preceding and/or following a JSON string, as is common with LLM responses, remove it and parse the JSON.

    Args:
        string: A string containing the JSON data and any preceding/following text.

    Returns:
        A dictionary of the parsed JSON, or None if it can't be parsed.
    """
    if not string:
        return None
    try:
        # Find the indices of the first '{' and the last '}'
        start_index = string.find("{")
        end_index = string.rfind("}")
        if start_index == -1 or end_index == -1:
            # No JSON found in the input string
            return None

        # Extract the JSON
        json_string = string[start_index : end_index + 1]
        json_object = json.loads(json_string)
        if not isinstance(json_object, dict):
            return None
        return json_object
    except json.JSONDecodeError:
        return None
