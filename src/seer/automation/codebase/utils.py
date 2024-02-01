import logging
import os
import shutil

from seer.automation.codebase.models import Document

logger = logging.getLogger(__name__)


def read_directory(
    path: str, extensions: list[str], parent_tmp_dir: str | None = None
) -> list[Document]:
    """
    Recursively reads all files in a directory that match the given list of file extensions and returns a Directory tree.

    :param directory: The directory to search in.
    :param extensions: A list of file extensions to include (e.g., ['.py', '.txt']).
    :return: A Directory object representing the directory tree with Document objects for files that match the given file extensions.
    """
    path_to_remove = parent_tmp_dir if parent_tmp_dir else path

    dir_children = []
    for entry in os.scandir(path):
        if entry.is_dir(follow_symlinks=False):
            dir_children.extend(read_directory(entry.path, extensions, path_to_remove))
        elif entry.is_file() and any(entry.name.endswith(ext) for ext in extensions):
            with open(entry.path, "r", encoding="utf-8") as f:
                text = f.read()

            truncated_path = entry.path.replace(path_to_remove, "")

            if truncated_path.startswith("/"):
                truncated_path = truncated_path[1:]

            dir_children.append(Document(path=truncated_path, text=text))
    return dir_children


def read_specific_files(repo_path: str, files: list[str]) -> list[Document]:
    """
    Reads the contents of specific files and returns a list of Document objects.

    :param files: A list of file paths to read.
    :return: A list of Document objects representing the file contents.
    """
    documents = []
    for file in files:
        with open(os.path.join(repo_path, file), "r", encoding="utf-8") as f:
            text = f.read()

        documents.append(Document(path=file, text=text))
    return documents


def cleanup_dir(directory: str):
    if os.path.exists(directory):
        shutil.rmtree(directory)
        logger.info(f"Cleaned up directory: {directory}")
    else:
        logger.info(f"Directory {directory} already cleaned!")
