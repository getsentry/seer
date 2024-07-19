import unittest

from seer.automation.agent.utils import extract_json_from_text, parse_json_with_keys


class TestParseJsonWithKeys(unittest.TestCase):
    def test_parse_json_with_valid_keys(self):
        json_str = '{"name": "John", "age": "30", "city": "New York"}'
        valid_keys = ["name", "age"]
        expected_output = {"name": "John", "age": "30"}
        assert parse_json_with_keys(json_str, valid_keys) == expected_output

    def test_parse_json_with_no_valid_keys(self):
        json_str = '{"name": "John", "age": "30", "city": "New York"}'
        valid_keys = ["country"]
        expected_output = {}
        assert parse_json_with_keys(json_str, valid_keys) == expected_output

    def test_parse_json_with_partial_valid_keys(self):
        json_str = '{"name": "John", "age": "30", "city": "New York"}'
        valid_keys = ["name", "city"]
        expected_output = {"name": "John", "city": "New York"}
        assert parse_json_with_keys(json_str, valid_keys) == expected_output

    def test_parse_json_with_merged_valid_keys(self):
        json_str = '{"name": "John", "age": "30", ", ": "New York"}'
        valid_keys = ["name", "age", "city"]
        expected_output = {"name": "John", "age": '30", ", ": "New York'}
        assert parse_json_with_keys(json_str, valid_keys) == expected_output

    def test_parse_json_with_lots_of_merged_valid_keys(self):
        json_str = '{"name": "John", "age": "30", ", ": ",":, ",": "New York"}'
        valid_keys = ["name", "age", "city"]
        expected_output = {"name": "John", "age": '30", ", ": ",":, ",": "New York'}
        assert parse_json_with_keys(json_str, valid_keys) == expected_output

    def test_parse_json_with_nested_json(self):
        json_str = (
            '{"name": "John", "age": "30", "address": {"city": "New York", "zipcode": "10021"}}'
        )
        valid_keys = ["name", "address"]
        expected_output = {"name": "John", "address": {"city": "New York", "zipcode": "10021"}}
        assert parse_json_with_keys(json_str, valid_keys) == expected_output

    def test_parse_json_still_works_with_invalid_json(self):
        json_str = '{"name": "John", "age": 30, "city": "New York"'
        valid_keys = ["name", "age", "city"]
        expected_output = {"name": "John", "age": 30, "city": "New York"}
        assert parse_json_with_keys(json_str, valid_keys) == expected_output

    def test_parse_json_with_non_string_values(self):
        json_str = '{"name": "John", "age": 30, "is_student": false}'
        valid_keys = ["name", "age", "is_student"]
        expected_output = {"name": "John", "age": 30, "is_student": False}
        assert parse_json_with_keys(json_str, valid_keys) == expected_output

        json_str = '{"name": "Doe", "grade": 90, "graduated": true}'
        valid_keys = ["name", "grade", "graduated"]
        expected_output = {"name": "Doe", "grade": 90, "graduated": True}
        assert parse_json_with_keys(json_str, valid_keys) == expected_output

    def test_process_json_str_with_newline(self):
        input_str = '{ "value": "Hello\\nWorld" }'
        valid_keys = ["value"]
        expected_output = {"value": "Hello\nWorld"}
        assert parse_json_with_keys(input_str, valid_keys) == expected_output

    def test_process_json_str_with_code_newline(self):
        input_str = '{ "value": "function sayHello() {\\n  console.log(\'Hello\\nWorld\');\\n}" }'
        valid_keys = ["value"]
        expected_output = {"value": "function sayHello() {\n  console.log('Hello\\nWorld');\n}"}
        assert parse_json_with_keys(input_str, valid_keys) == expected_output


class TestExtractJsonFromText(unittest.TestCase):
    def test_valid_json_with_surrounding_text(self):
        input_string = 'Some text before {"key": "value"} and some text after'
        expected_json = {"key": "value"}
        result = extract_json_from_text(input_string)
        assert result == expected_json

    def test_valid_json_with_preceding_text(self):
        input_string = 'Some text before {"key": "value"}'
        expected_json = {"key": "value"}
        result = extract_json_from_text(input_string)
        assert result == expected_json

    def test_valid_json_with_following_text(self):
        input_string = '{"key": "value"} and some text after'
        expected_json = {"key": "value"}
        result = extract_json_from_text(input_string)
        assert result == expected_json

    def test_valid_json_without_surrounding_text(self):
        input_string = '{"key": "value"}'
        expected_json = {"key": "value"}
        result = extract_json_from_text(input_string)
        assert result == expected_json

    def test_nested_json(self):
        input_string = 'Text before {"outer": {"inner": "value"}} text after'
        expected_json = {"outer": {"inner": "value"}}
        result = extract_json_from_text(input_string)
        assert result == expected_json

    def test_no_json(self):
        input_string = "This is a string without any JSON"
        result = extract_json_from_text(input_string)
        assert result is None

    def test_invalid_json(self):
        input_string = 'Text before {"key": "value" text after'
        result = extract_json_from_text(input_string)
        assert result is None

    def test_empty_string(self):
        input_string = ""
        result = extract_json_from_text(input_string)
        assert result is None

    def test_no_input(self):
        input_string = None
        result = extract_json_from_text(input_string)
        assert result is None
