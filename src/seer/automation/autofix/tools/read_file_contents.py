import logging
import os

import sentry_sdk

from seer.automation.autofix.tools.tools import observe

logger = logging.getLogger(__name__)


@observe(name="Read File Contents from File System")
@sentry_sdk.trace
def read_file_contents(repo_dir: str, file_path: str) -> tuple[str | None, str | None]:
    path = os.path.join(repo_dir, file_path)

    if not os.path.exists(path):
        logger.warning(f"File does not exist: {file_path}")
        return None, f"Error: File {file_path} does not exist"

    if not os.path.isfile(path):
        logger.warning(f"Path exists but is not a file: {file_path}")
        return None, f"Error: Path {file_path} exists but is not a file"

    try:
        with open(path, "r") as f:
            return f.read(), None
    except Exception as e:
        logger.exception(f"Error reading file {file_path}: {e}")
        return None, f"Error reading file {file_path}: {e}"
