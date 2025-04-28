import pytest
from unittest.mock import patch, MagicMock
from seer.automation.codebase.repo_client import RepoClient, RepoClientType, get_read_app_credentials, get_write_app_credentials, get_codecov_unit_test_app_credentials
from seer.automation.models import RepoDefinition

@pytest.fixture
def mock_repo_definition():
    return RepoDefinition(
        full_name="test/repo",
        default_branch="main",
        clone_url="https://github.com/test/repo.git",
        ssh_url="git@github.com:test/repo.git",
        external_id="123456",
    )

def test_repo_client_type_enum():
    assert RepoClientType.READ == "read"
    assert RepoClientType.WRITE == "write"
    assert RepoClientType.CODECOV_UNIT_TEST == "codecov_unit_test"

@patch('seer.automation.codebase.repo_client.get_read_app_credentials')
@patch('seer.automation.codebase.repo_client.get_write_app_credentials')
@patch('seer.automation.codebase.repo_client.get_codecov_unit_test_app_credentials')
def test_repo_client_from_repo_definition(mock_codecov_creds, mock_write_creds, mock_read_creds, mock_repo_definition):
    mock_read_creds.return_value = ("read_app_id", "read_private_key")
    mock_write_creds.return_value = ("write_app_id", "write_private_key")
    mock_codecov_creds.return_value = ("codecov_app_id", "codecov_private_key")

    # Test READ type
    client = RepoClient.from_repo_definition(mock_repo_definition, RepoClientType.READ)
    mock_read_creds.assert_called_once()
    assert client.github_auth.app_id == "read_app_id"
    assert client.github_auth.private_key == "read_private_key"

    # Test WRITE type
    client = RepoClient.from_repo_definition(mock_repo_definition, RepoClientType.WRITE)
    mock_write_creds.assert_called_once()
    assert client.github_auth.app_id == "write_app_id"
    assert client.github_auth.private_key == "write_private_key"

    # Test CODECOV_UNIT_TEST type
    client = RepoClient.from_repo_definition(mock_repo_definition, RepoClientType.CODECOV_UNIT_TEST)
    mock_codecov_creds.assert_called_once()
    assert client.github_auth.app_id == "codecov_app_id"
    assert client.github_auth.private_key == "codecov_private_key"

@patch('seer.automation.codebase.repo_client.Auth')
@patch('seer.automation.codebase.repo_client.Github')
def test_repo_client_init(mock_github, mock_auth, mock_repo_definition):
    mock_auth.AppInstallationAuth.return_value = MagicMock()
    mock_github.return_value = MagicMock()

    client = RepoClient("app_id", "private_key", mock_repo_definition)

    mock_auth.AppInstallationAuth.assert_called_once_with(
        app_id="app_id",
        private_key="private_key",
        installation_id=mock_repo_definition.external_id,
    )
    mock_github.assert_called_once_with(auth=mock_auth.AppInstallationAuth.return_value)
    assert client.repo_name == mock_repo_definition.full_name

@patch('seer.automation.codebase.repo_client.AppConfig')
def test_get_read_app_credentials(mock_app_config):
    mock_app_config.return_value.GITHUB_READ_APP_ID = "read_app_id"
    mock_app_config.return_value.GITHUB_READ_PRIVATE_KEY = "read_private_key"

    app_id, private_key = get_read_app_credentials()
    assert app_id == "read_app_id"
    assert private_key == "read_private_key"

@patch('seer.automation.codebase.repo_client.AppConfig')
def test_get_write_app_credentials(mock_app_config):
    mock_app_config.return_value.GITHUB_WRITE_APP_ID = "write_app_id"
    mock_app_config.return_value.GITHUB_WRITE_PRIVATE_KEY = "write_private_key"

    app_id, private_key = get_write_app_credentials()
    assert app_id == "write_app_id"
    assert private_key == "write_private_key"

@patch('seer.automation.codebase.repo_client.AppConfig')
def test_get_codecov_unit_test_app_credentials(mock_app_config):
    mock_app_config.return_value.GITHUB_CODECOV_UNIT_TEST_APP_ID = "codecov_app_id"
    mock_app_config.return_value.GITHUB_CODECOV_UNIT_TEST_PRIVATE_KEY = "codecov_private_key"

    app_id, private_key = get_codecov_unit_test_app_credentials()
    assert app_id == "codecov_app_id"
    assert private_key == "codecov_private_key"