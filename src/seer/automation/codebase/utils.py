import functools
import logging
import os
import shutil

from seer.automation.codebase.models import Document
from seer.automation.models import StacktraceFrame

logger = logging.getLogger(__name__)

language_to_extensions = {
    "bash": [".sh"],
    "c": [".c"],
    "c_sharp": [".cs"],
    "commonlisp": [".lisp"],
    "cpp": [".cpp", ".cxx", ".cc", ".hpp"],
    "css": [".css", ".scss", ".sass"],
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
def get_all_supported_extensions() -> set[str]:
    """
    Returns a set of all supported file extensions across all languages.

    :return: A set of file extensions including the dot prefix (e.g. {'.py', '.js'})
    """
    extensions = set()
    for extensions_list in language_to_extensions.values():
        extensions.update(extensions_list)
    return extensions


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


def read_specific_files(repo_path: str, files: list[str] | set[str]) -> list[Document]:
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

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            logger.warning(f"Unicode decode error: {file_path}")
            continue
        except FileNotFoundError:
            logger.warning(f"File not found: {file_path}")
            documents.append(Document(path=file, text="", language=language))
            continue

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

    filename = frame.filename or frame.package
    if filename:
        # Remove leading './' or '.' from filename
        filename = filename.lstrip("./")
        frame_split = filename.split("/")[::-1]

        if len(src_split) > 0 and len(frame_split) > 0 and len(src_split) >= len(frame_split):
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
