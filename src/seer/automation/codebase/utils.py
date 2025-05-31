import datetime
import functools
import logging
import os
import shutil
import time
from pathlib import Path

from seer.automation.codebase.models import Document
from seer.automation.models import StacktraceFrame, right_justified

logger = logging.getLogger(__name__)

language_to_extensions = {
    "apl": [".apl"],
    "apex": [".cls", ".apex"],
    "assembly": [".asm", ".s"],
    "astro": [".astro"],
    "bash": [".sh"],
    "bicep": [".bicep"],
    "c": [".c", ".d"],
    "cmake": [".cmake"],
    "c_sharp": [".cs"],
    "clojure": [".clj", ".cljs"],
    "coffeescript": [".coffee"],
    "commonlisp": [".lisp"],
    "conf": [".conf", ".cnf", ".cfg"],
    "cpp": [".cpp", ".cxx", ".cc", ".hpp"],
    "crystal": [".cr"],
    "css": [".css", ".scss", ".sass"],
    "cuda": [".cu", ".cuh"],
    "dart": [".dart"],
    "dockerfile": ["Dockerfile"],
    "dot": [".dot"],
    "elisp": [".el"],
    "elixir": [".ex", ".exs"],
    "elm": [".elm"],
    "embedded_template": [".tmpl", ".tpl", ".ejs"],
    "erlang": [".erl", ".hrl"],
    "fixed_form_fortran": [".f", ".for", ".f77"],
    "fortran": [".f90", ".f95", ".f03", ".f08"],
    "fsharp": [".fs", ".fsx"],
    "gdscript": [".gd", ".gdshader"],
    "gleam": [".gleam"],
    "glsl": [".glsl", ".frag", ".vert"],
    "go": [".go"],
    "gradle": [".gradle"],
    "graphql": [".graphql", ".gql"],
    "groovy": [".groovy", ".gvy"],
    "hack": [".hack"],
    "handlebars": [".hbs", ".handlebars"],
    "haskell": [".hs", ".lhs"],
    "hcl": [".hcl"],
    "hlsl": [".hlsl"],
    "html": [".html", ".htm"],
    "ignore": [".gitignore", ".npmignore"],
    "ini": [".ini"],
    "java": [".java"],
    "javascript": [".js", ".jsx"],
    "jest": [".jest.js", ".spec.js", ".test.js"],
    "jsdoc": [".jsdoc"],
    "json": [".json"],
    "julia": [".jl"],
    "jupyter": [".ipynb"],
    "kotlin": [".kt", ".kts"],
    "less": [".less"],
    "lua": [".lua"],
    "make": ["Makefile"],
    "markdown": [".md", ".markdown", ".mdx"],
    "matlab": [".m", ".mat"],
    "mojo": [".🔥", ".mojo"],
    "nim": [".nim"],
    "nix": [".nix"],
    "objc": [".m", ".h", ".mm"],
    "ocaml": [".ml", ".mli"],
    "pascal": [".pas", ".pp"],
    "pegjs": [".pegjs"],
    "perl": [".pl", ".pm"],
    "php": [".php", ".phtml", ".php3", ".php4", ".php5", ".php7", ".phps", ".php-s"],
    "powershell": [".ps1", ".psm1"],
    "proto": [".proto"],
    "pug": [".pug", ".jade"],
    "puppet": [".pp"],
    "purescript": [".purs"],
    "python": [".py"],
    "ql": [".ql"],
    "r": [".r", ".R"],
    "racket": [".rkt"],
    "raku": [".raku", ".rakumod"],
    "regex": [".re"],
    "rescript": [".res", ".resi"],
    "rst": [".rst"],
    "ruby": [".rb"],
    "rust": [".rs"],
    "scala": [".scala", ".sc"],
    "shell": [".sh", ".bash", ".zsh"],
    "smali": [".smali"],
    "solidity": [".sol"],
    "sql": [".sql"],
    "sqlite": [".sqlite"],
    "stylus": [".styl"],
    "svelte": [".svelte"],
    "swift": [".swift"],
    "terraform": [".tf", ".tfvars"],
    "text": [".txt", ".rtf"],
    "toml": [".toml"],
    "tsq": [".tsq"],
    "tsx": [".tsx"],
    "twig": [".twig"],
    "typescript": [".ts"],
    "typescriptdef": [".d.ts"],
    "v": [".v"],
    "verilog": [".v", ".vh"],
    "vhdl": [".vhd", ".vhdl"],
    "vue": [".vue"],
    "wasm": [".wat", ".wasm"],
    "wgsl": [".wgsl"],
    "xml": [".xml", ".xaml"],
    "yaml": [".yaml", ".yml"],
    "zig": [".zig"],
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


def cleanup_dir(directory: str, max_retries: int = 3, initial_delay: float = 0.5):
    """
    Clean up a directory with retries on failure.

    Args:
        directory: The directory to remove
        max_retries: Maximum number of retries before giving up
        initial_delay: Initial delay between retries, doubles after each retry
    """
    if not os.path.exists(directory):
        logger.info(f"Directory {directory} already cleaned!")
        return

    delay = initial_delay
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            shutil.rmtree(directory)
            logger.info(f"Cleaned up directory: {directory}")
            return
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                logger.warning(
                    f"Failed to clean directory {directory} (attempt {attempt + 1}/{max_retries + 1}): {e}"
                )
                time.sleep(delay)
                delay *= 2
            else:
                raise OSError(
                    f"Failed to clean directory {directory} after {max_retries + 1} attempts"
                ) from last_error


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


def code_snippet(
    lines: list[str],
    start_line: int,
    end_line: int,
    padding_size: int = 0,
    start_line_override: int | None = None,
) -> list[str]:
    """
    `start_line` and `end_line` are assumed to be 1-indexed. `end_line` is inclusive.

    Pass `start_line_override` to override the line numbers as shown in the snippet.
    """
    if (start_line <= 0) or (end_line <= 0):
        raise ValueError("start_line and end_line must be greater than 0. They're 1-indexed.")

    start_idx = start_line - 1
    end_idx = end_line - 1
    start_snippet = max(0, start_idx - padding_size)
    end_snippet = min(end_idx + padding_size + 1, len(lines))
    lines_snippet = lines[start_snippet:end_snippet]

    start_line = start_line_override or start_snippet + 1
    end_line = start_line + len(lines_snippet) - 1
    line_numbers = right_justified(start_line, end_line)
    return [f"{line_number}| {line}" for line_number, line in zip(line_numbers, lines_snippet)]


def left_truncated_paths(path: Path, max_num_paths: int = 2) -> list[str]:
    """
    Example::

        path = Path("src/seer/automation/agent/client.py")
        paths = left_truncated_paths(path, 2)
        assert paths == [
            "seer/automation/agent/client.py",
            "automation/agent/client.py",
        ]
    """
    parts = list(path.parts)
    num_dirs = len(parts) - 1  # -1 for the filename
    num_paths = min(max_num_paths, num_dirs)

    result = []
    for _ in range(num_paths):
        parts.pop(0)
        result.append(Path(*parts).as_posix())
    return result


def ensure_timezone_aware(dt: datetime.datetime | None) -> datetime.datetime | None:
    """
    Ensures a datetime is timezone-aware by adding UTC timezone if it's naive.
    Returns None if the input is None.

    Args:
        dt: The datetime object to ensure is timezone-aware

    Returns:
        A timezone-aware datetime object, or None if dt is None
    """
    if dt is None:
        return None

    if dt.tzinfo is None:
        return dt.replace(tzinfo=datetime.UTC)
    return dt
