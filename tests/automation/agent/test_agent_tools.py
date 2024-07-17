from unittest.mock import Mock, patch

import pytest

from seer.automation.agent.tools import FunctionTool, get_full_exception_string


class TestGetFullExceptionString:
    def test_simple_exception(self):
        exc = ValueError("Simple error")
        assert get_full_exception_string(exc) == "Simple error"

    def test_chained_exception(self):
        try:
            raise RuntimeError("Main error") from ValueError("Root cause")
        except RuntimeError as exc:
            assert (
                get_full_exception_string(exc)
                == "Main error\n\nThe above exception was the direct cause of the following exception:\n\nRoot cause"
            )

    def test_empty_main_exception(self):
        try:
            raise RuntimeError() from ValueError("Root cause")
        except RuntimeError as exc:
            assert get_full_exception_string(exc) == "Root cause"


class TestFunctionTool:
    @pytest.fixture
    def mock_function(self):
        return Mock(return_value="Success")

    @pytest.fixture
    def function_tool(self, mock_function):
        return FunctionTool(
            name="test_tool",
            description="A test tool",
            fn=mock_function,
            parameters=[{"name": "param1", "type": "string"}],
        )

    def test_successful_call(self, function_tool):
        result = function_tool.call(param1="test")
        assert result == "Success"

    def test_exception_handling(self, function_tool):
        function_tool.fn.side_effect = ValueError("Test error")

        with patch("seer.automation.agent.tools.logger") as mock_logger:
            result = function_tool.call(param1="test")

        assert result.startswith("Error: Test error")
        mock_logger.exception.assert_called_once()

    def test_chained_exception_handling(self, function_tool):
        cause = ValueError("Root cause")
        main_error = RuntimeError("Main error")
        main_error.__cause__ = cause
        function_tool.fn.side_effect = main_error

        with patch("seer.automation.agent.tools.logger") as mock_logger:
            result = function_tool.call(param1="test")

        expected = "Error: Main error\n\nThe above exception was the direct cause of the following exception:\n\nRoot cause"
        assert result == expected
        mock_logger.exception.assert_called_once()

    @pytest.mark.parametrize(
        "model, expected",
        [
            (
                "gpt",
                {
                    "type": "function",
                    "function": {
                        "name": "test_tool",
                        "description": "A test tool",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "param1": {
                                    "type": "string",
                                    "description": "",
                                }
                            },
                            "required": [],
                        },
                    },
                },
            ),
            (
                "claude",
                {
                    "name": "test_tool",
                    "description": "A test tool",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "param1": {
                                "type": "string",
                                "description": "",
                            }
                        },
                        "required": [],
                    },
                },
            ),
        ],
    )
    def test_to_dict(self, function_tool, model, expected):
        assert function_tool.to_dict(model=model) == expected
