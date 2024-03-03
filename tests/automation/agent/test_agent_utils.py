import unittest

from seer.automation.agent.utils import parse_json_with_keys


class TestParseJsonWithKeys(unittest.TestCase):
    def test_parse_json_with_valid_keys(self):
        json_str = '{"name": "John", "age": "30", "city": "New York"}'
        valid_keys = ["name", "age"]
        expected_output = {"name": "John", "age": "30"}
        self.assertEqual(parse_json_with_keys(json_str, valid_keys), expected_output)

    def test_parse_json_with_no_valid_keys(self):
        json_str = '{"name": "John", "age": "30", "city": "New York"}'
        valid_keys = ["country"]
        expected_output = {}
        self.assertEqual(parse_json_with_keys(json_str, valid_keys), expected_output)

    def test_parse_json_with_partial_valid_keys(self):
        json_str = '{"name": "John", "age": "30", "city": "New York"}'
        valid_keys = ["name", "city"]
        expected_output = {"name": "John", "city": "New York"}
        self.assertEqual(parse_json_with_keys(json_str, valid_keys), expected_output)

    def test_parse_json_with_merged_valid_keys(self):
        json_str = '{"name": "John", "age": "30", ", ": "New York"}'
        valid_keys = ["name", "age", "city"]
        expected_output = {"name": "John", "age": '30", ", ": "New York'}
        self.assertEqual(parse_json_with_keys(json_str, valid_keys), expected_output)

    def test_parse_json_with_lots_of_merged_valid_keys(self):
        json_str = '{"name": "John", "age": "30", ", ": ",":, ",": "New York"}'
        valid_keys = ["name", "age", "city"]
        expected_output = {"name": "John", "age": '30", ", ": ",":, ",": "New York'}
        self.assertEqual(parse_json_with_keys(json_str, valid_keys), expected_output)

    def test_parse_json_with_nested_json(self):
        json_str = (
            '{"name": "John", "age": "30", "address": {"city": "New York", "zipcode": "10021"}}'
        )
        valid_keys = ["name", "address"]
        expected_output = {"name": "John", "address": {"city": "New York", "zipcode": "10021"}}
        self.assertEqual(parse_json_with_keys(json_str, valid_keys), expected_output)

    def test_parse_json_still_works_with_invalid_json(self):
        json_str = '{"name": "John", "age": 30, "city": "New York"'
        valid_keys = ["name", "age", "city"]
        expected_output = {"name": "John", "age": 30, "city": "New York"}
        self.assertEqual(parse_json_with_keys(json_str, valid_keys), expected_output)

    def test_parse_json_with_non_string_values(self):
        json_str = '{"name": "John", "age": 30, "is_student": false}'
        valid_keys = ["name", "age", "is_student"]
        expected_output = {"name": "John", "age": 30, "is_student": False}
        self.assertEqual(parse_json_with_keys(json_str, valid_keys), expected_output)

        json_str = '{"name": "Doe", "grade": 90, "graduated": true}'
        valid_keys = ["name", "grade", "graduated"]
        expected_output = {"name": "Doe", "grade": 90, "graduated": True}
        self.assertEqual(parse_json_with_keys(json_str, valid_keys), expected_output)

    def test_process_json_str_with_newline(self):
        input_str = '{ "value": "Hello\\nWorld" }'
        valid_keys = ["value"]
        expected_output = {"value": "Hello\nWorld"}
        self.assertEqual(parse_json_with_keys(input_str, valid_keys), expected_output)

    def test_process_json_str_with_code_newline(self):
        input_str = '{ "value": "function sayHello() {\\n  console.log(\'Hello\\nWorld\');\\n}" }'
        valid_keys = ["value"]
        expected_output = {"value": "function sayHello() {\n  console.log('Hello\\nWorld');\n}"}
        self.assertEqual(parse_json_with_keys(input_str, valid_keys), expected_output)
