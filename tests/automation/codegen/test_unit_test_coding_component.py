import unittest
from unittest.mock import MagicMock

from seer.automation.codebase.repo_client import RepoClientType
from seer.automation.codegen.codegen_context import CodegenContext
from seer.automation.codegen.unit_test_coding_component import UnitTestCodingComponent


class TestUnitTestCodingComponent(unittest.TestCase):
    def setUp(self):
        self.mock_context = MagicMock(spec=CodegenContext)
        self.component = UnitTestCodingComponent(self.mock_context)

    def test_get_client_type_with_codecov_request(self):
        client_type = self.component._get_client_type(is_codecov_request=True)
        self.assertEqual(client_type, RepoClientType.CODECOV_PR_REVIEW)

    def test_get_client_type_without_codecov_request(self):
        client_type = self.component._get_client_type(is_codecov_request=False)
        self.assertEqual(client_type, RepoClientType.CODECOV_UNIT_TEST)
