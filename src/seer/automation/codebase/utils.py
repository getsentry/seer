import functools
import logging
import os
import shutil

from seer.automation.autofix.models import StacktraceFrame
from seer.automation.codebase.models import Document

logger = logging.getLogger(__name__)

language_to_extensions = {
    "bash": [".sh"],
    "c": [".c"],
    "c_sharp": [".cs"],
    "commonlisp": [".lisp"],
    "cpp": [".cpp", ".cxx", ".cc", ".hpp"],
    "css": [".css"],
    "dockerfile": ["Dockerfile"],
    "dot": [".dot"],
    "elisp": [".el"],
    "elixir": [".ex", ".exs"],
    "elm": [".elm"],
    "embedded_template": [".tmpl", ".tpl"],
    "erlang": [".erl", ".hrl"],
    "fixed_form_fortran": [".f", ".for", ".f77"],
    "fortran": [".f90", ".f95", ".f03", ".f08"],
    "go": [".go"],
    "hack": [".hack"],
    "haskell": [".hs", ".lhs"],
    "hcl": [".hcl"],
    "html": [".html", ".htm"],
    "java": [".java"],
    "javascript": [".js", "jsx"],
    "jsdoc": [".jsdoc"],
    "json": [".json"],
    "julia": [".jl"],
    "kotlin": [".kt", ".kts"],
    "lua": [".lua"],
    "make": ["Makefile"],
    "markdown": [".md", ".markdown"],
    "objc": [".m", ".h"],
    "ocaml": [".ml", ".mli"],
    "perl": [".pl", ".pm"],
    "php": [".php", ".phtml", ".php3", ".php4", ".php5", ".php7", ".phps", ".php-s"],
    "python": [".py"],
    "ql": [".ql"],
    "r": [".r", ".R"],
    "regex": [".re"],
    "rst": [".rst"],
    "ruby": [".rb"],
    "rust": [".rs"],
    "scala": [".scala", ".sc"],
    "sql": [".sql"],
    "sqlite": [".sqlite"],
    "toml": [".toml"],
    "tsq": [".tsq"],
    "tsx": [".tsx"],
    "typescript": [".ts"],
    "yaml": [".yaml", ".yml"],
}


@functools.cache
def get_extension_to_language_map():
    extension_to_language: dict[str, str] = {}
    for language, extensions in language_to_extensions.items():
        for extension in extensions:
            extension_to_language[extension] = language
    return extension_to_language


def get_language_from_path(path: str) -> str | None:
    extension = os.path.splitext(path)[1]
    return get_extension_to_language_map().get(extension, None)


def read_directory(
    path: str,
    parent_tmp_dir: str | None = None,
    max_file_size=2 * 1024 * 1024,  # 2 MB
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
            dir_children.extend(read_directory(entry.path, path_to_remove))
        elif entry.is_file() and entry.stat().st_size < max_file_size:
            language = get_language_from_path(entry.path)

            # TODO: Support languages that are out of this list in the near future by simply using dumb chunking.
            if not language:
                continue

            with open(entry.path, "r", encoding="utf-8") as f:
                text = f.read()

            truncated_path = entry.path.replace(path_to_remove, "")

            if truncated_path.startswith("/"):
                truncated_path = truncated_path[1:]

            dir_children.append(Document(path=truncated_path, text=text, language=language))
    return dir_children


def read_specific_files(repo_path: str, files: list[str]) -> list[Document]:
    """
    Reads the contents of specific files and returns a list of Document objects.

    :param files: A list of file paths to read.
    :return: A list of Document objects representing the file contents.
    """
    documents = []
    for file in files:
        file_path = os.path.join(repo_path, file)

        language = get_language_from_path(file_path)

        # TODO: Support languages that are out of this list in the near future by simply using dumb chunking.
        if not language:
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        documents.append(Document(path=file, text=text, language=language))
    return documents


def cleanup_dir(directory: str):
    if os.path.exists(directory):
        shutil.rmtree(directory)
        logger.info(f"Cleaned up directory: {directory}")
    else:
        logger.info(f"Directory {directory} already cleaned!")


def potential_frame_match(src_file: str, frame: StacktraceFrame) -> bool:
    """Determine if the frame filename represents a source code file."""
    match = False

    src_split = src_file.split("/")[::-1]
    frame_split = frame.filename.split("/")[::-1]

    if len(src_split) > 1 and len(frame_split) > 1 and len(src_split) >= len(frame_split):
        for i in range(len(frame_split)):
            if src_split[i] == frame_split[i]:
                match = True
            else:
                match = False
                break

    return match


def group_documents_by_language(documents: list[Document]) -> dict[str, list[Document]]:
    file_type_count: dict[str, list[Document]] = {}
    for doc in documents:
        file_type = doc.language
        if file_type not in file_type_count:
            file_type_count[file_type] = []
        file_type_count[file_type].append(doc)

    return file_type_count
