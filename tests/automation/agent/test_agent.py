import unittest
from unittest.mock import MagicMock

from seer.automation.agent.agent import GptAgent
from seer.automation.agent.models import ToolCall
from seer.automation.agent.tools import FunctionTool


class TestGptAgentCallToolIntegration(unittest.TestCase):
    def setUp(self):
        self.agent = GptAgent()

    def test_call_tool(self):
        # Setup mock tool and agent
        mock_fn = MagicMock(return_value="Tool called successfully")

        self.agent.tools = [
            FunctionTool(
                name="mock_tool",
                description="tool",
                fn=mock_fn,
                parameters=[
                    {"name": "arg1", "type": "str"},
                    {"name": "arg2", "type": "str"},
                    {"name": "arg3", "type": "int"},
                ],
            )
        ]

        tool_call = ToolCall(
            id="1",
            function="mock_tool",
            args='{"arg1": "value1\\nbar(\'\\n\')", "arg2": "value2", "arg3": 123}',
        )

        # Call the method
        result = self.agent.call_tool(tool_call)

        # Assertions
        self.assertEqual(result.content, "Tool called successfully")
        self.assertEqual(result.role, "tool")
        self.assertEqual(result.tool_call_id, "1")
        mock_fn.assert_called_once_with(arg1="value1\nbar('\\n')", arg2="value2", arg3=123)
