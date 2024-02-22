import functools
import os
from typing import Any

import torch
import tree_sitter_languages
from sentence_transformers import SentenceTransformer


def get_torch_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@functools.cache
def get_embedding_model():
    model = SentenceTransformer(
        os.path.join("./", "models", "autofix_embeddings_v0"),
        trust_remote_code=True,
    ).to(get_torch_device())

    model.max_seq_length = 4096

    return model


def parse_json_with_keys(json_str: str, valid_keys: list[str]) -> dict[str, Any]:
    """
    NOTE: This function will only work if the json is not nested, and the values are all strings.

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
                if (
                    key_contains_invalid_chars
                    and node.prev_named_sibling
                    and node.prev_named_sibling.type == "pair"
                ):
                    correct_key = node.prev_named_sibling.children[0].text[1:-1].decode("utf-8")
                    is_previous_child_str = node.prev_named_sibling.children[2].type == "string"
                    start_byte = (
                        node.prev_named_sibling.children[2].start_byte + 1
                        if is_previous_child_str
                        else node.prev_named_sibling.children[2].start_byte
                    )
                    is_current_child_str = node.children[2].type == "string"
                    end_byte = (
                        node.children[2].end_byte - 1
                        if is_current_child_str
                        else node.children[2].end_byte
                    )
                    parsed_json.append(
                        (correct_key, tree.root_node.text[start_byte:end_byte].decode("utf-8"))
                    )
            else:
                value = (
                    node.children[2].text[1:-1].decode("utf-8")
                    if node.children[2].type == "string"
                    else node.children[2].text.decode("utf-8")
                )
                parsed_json.append((key, value))

    json_dict = {}
    for key, value in parsed_json:
        json_dict[key] = value

    return json_dict
