import unittest

from seer.automation.utils import parse_json_with_keys


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
        expected_output = {"name": "John", "address": '{"city": "New York", "zipcode": "10021"}'}
        self.assertEqual(parse_json_with_keys(json_str, valid_keys), expected_output)

    def test_parse_json_still_works_with_invalid_json(self):
        json_str = '{"name": "John", "age": 30, "city": "New York"'
        valid_keys = ["name", "age", "city"]
        expected_output = {"name": "John", "age": "30", "city": "New York"}
        self.assertEqual(parse_json_with_keys(json_str, valid_keys), expected_output)


if __name__ == "__main__":
    unittest.main()
