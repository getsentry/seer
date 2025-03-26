import unittest

from seer.automation.models import RepoDefinition


class TestRepoDefinition(unittest.TestCase):
    def test_repo_definition_basic_fields(self):
        """Test basic fields of RepoDefinition"""
        repo = RepoDefinition(
            owner="test-owner",
            name="test-repo",
            external_id="test-external-id",
            provider="github",
        )

        self.assertEqual(repo.owner, "test-owner")
        self.assertEqual(repo.name, "test-repo")
        self.assertEqual(repo.external_id, "test-external-id")
        self.assertEqual(repo.provider, "github")
        self.assertIsNone(repo.branch_name)
        self.assertIsNone(repo.instructions)
        self.assertIsNone(repo.base_commit_sha)
        self.assertEqual(repo.provider_raw, "github")
        self.assertEqual(repo.full_name, "test-owner/test-repo")

    def test_repo_definition_all_fields(self):
        """Test all fields of RepoDefinition including optional ones"""
        repo = RepoDefinition(
            owner="test-owner",
            name="test-repo",
            external_id="test-external-id",
            provider="github",
            branch_name="feature-branch",
            instructions="# Custom instructions",
            base_commit_sha="abcdef123456",
            provider_raw="github",
        )

        self.assertEqual(repo.owner, "test-owner")
        self.assertEqual(repo.name, "test-repo")
        self.assertEqual(repo.external_id, "test-external-id")
        self.assertEqual(repo.provider, "github")
        self.assertEqual(repo.branch_name, "feature-branch")
        self.assertEqual(repo.instructions, "# Custom instructions")
        self.assertEqual(repo.base_commit_sha, "abcdef123456")
        self.assertEqual(repo.provider_raw, "github")
        self.assertEqual(repo.full_name, "test-owner/test-repo")

    def test_repo_definition_property_full_name(self):
        """Test the full_name property of RepoDefinition"""
        repo = RepoDefinition(
            owner="different-owner",
            name="different-repo",
            external_id="test-id",
            provider="github",
        )
        self.assertEqual(repo.full_name, "different-owner/different-repo")

    def test_repo_definition_model_dump(self):
        """Test the model_dump method of RepoDefinition"""
        repo = RepoDefinition(
            owner="test-owner",
            name="test-repo",
            external_id="test-external-id",
            provider="github",
            branch_name="main",
            instructions="# Test instructions",
        )

        dumped = repo.model_dump()
        self.assertIsInstance(dumped, dict)
        self.assertEqual(dumped["owner"], "test-owner")
        self.assertEqual(dumped["name"], "test-repo")
        self.assertEqual(dumped["external_id"], "test-external-id")
        self.assertEqual(dumped["provider"], "github")
        self.assertEqual(dumped["branch_name"], "main")
        self.assertEqual(dumped["instructions"], "# Test instructions")
        self.assertIsNone(dumped["base_commit_sha"])
        self.assertEqual(dumped["provider_raw"], "github")
