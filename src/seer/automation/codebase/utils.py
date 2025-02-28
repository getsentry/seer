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


def potential_frame_match(src_file: str, frame: StacktraceFrame) -> tuple[bool, float]:
 """
 Determine if the frame filename represents a source code file.
 Returns a tuple of (match_found, confidence_score) where confidence_score is a value from 0.0 to 1.0
 indicating how confident we are in the match.
 """
 # Normalize paths for comparison
 def normalize_path(path):
 if not path:
 return ""
 # Strip leading './' and '/'
 path = path.lstrip("./").lstrip("/")
 # Convert to lowercase for case-insensitive comparison
 return path.lower()
 
 src_normalized = normalize_path(src_file)
 frame_path = frame.filename or frame.package
 frame_normalized = normalize_path(frame_path)
 
 if not frame_normalized:
 return False, 0.0
 
 # Quick exact match check
 if src_normalized == frame_normalized:
 return True, 1.0
 
 # Component-wise matching (from the end)
 src_components = src_normalized.split('/')
 frame_components = frame_normalized.split('/')
 
 # File name matching (highest priority)
 if src_components and frame_components and src_components[-1] == frame_components[-1]:
 # Filename matches are a good sign
 base_score = 0.6
 else:
 # If filenames don't match, lower starting score
 base_score = 0.3
 
 # Check for path suffix match (e.g., "src/module/file.py" matches "module/file.py")
 max_components = min(len(src_components), len(frame_components))
 matching_components = 0
 
 for i in range(1, max_components + 1):
 if src_components[-i] == frame_components[-i]:
 matching_components += 1
 else:
 break
 
 if matching_components == 0:
 return False, 0.0
 
 # Calculate score based on matching components
 component_score = matching_components / max(len(src_components), len(frame_components))
 
 # Check if one path is contained in the other (lower priority, but still useful)
 containment_score = 0.0
 if src_normalized in frame_normalized or frame_normalized in src_normalized:
 containment_score = 0.2
 
 # Combine scores with appropriate weighting
 final_score = base_score * 0.5 + component_score * 0.4 + containment_score * 0.1
 
 # Only return true if we have a reasonable confidence
 return final_score >= 0.4, final_score


def group_documents_by_language(documents: list[Document]) -> dict[str, list[Document]]:
    file_type_count: dict[str, list[Document]] = {}
    for doc in documents:
        file_type = doc.language
        if file_type not in file_type_count:
            file_type_count[file_type] = []
        file_type_count[file_type].append(doc)

    return file_type_count
