"""@ file reference resolution — scan user text for @path tokens and resolve to file contents."""

from __future__ import annotations

from typing import NamedTuple

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document


class FileContext(NamedTuple):
    """Resolved file reference."""
    path: str      # display path (relative to cwd when possible)
    content: str   # file contents as UTF-8 text


class AtFileError(Exception):
    """Base exception for @ file reference errors."""


class FileNotFoundAtError(AtFileError):
    """@ referenced file does not exist."""


class GlobNoMatchError(AtFileError):
    """Glob pattern in @ reference matched no files."""


class FileTooLargeError(AtFileError):
    """@ referenced file exceeds size limit."""


class BinaryFileError(AtFileError):
    """@ referenced file appears to be binary."""

import re

# Matches inline `...` and fenced ```...``` blocks (non-greedy)
_BACKTICK_BLOCK = re.compile(r'```[\s\S]*?```|`[^`\n]+`')
# Same pattern but capturing for re.split to include matched blocks in result
_BACKTICK_SPLIT = re.compile(r'(```[\s\S]*?```|`[^`\n]+`)')

# Matches @ followed by one or more non-whitespace chars
_AT_REF = re.compile(r'@(\S+)')


def _scan_for_at_refs(text: str) -> list[str]:
    """Find @path references in text, ignoring any inside backtick blocks."""
    safe = _BACKTICK_BLOCK.sub('', text)
    return [m.group(1) for m in _AT_REF.finditer(safe)]


from pathlib import Path

_GLOB_CHARS = frozenset('*?[')


def _expand_paths(refs: list[str], cwd: str) -> list[Path]:
    """Resolve @ references to absolute Paths. Expands globs, deduplicates."""
    base = Path(cwd).resolve()
    seen: set[Path] = set()
    result: list[Path] = []

    for ref in refs:
        p = Path(ref)
        if not p.is_absolute():
            p = base / p

        if _GLOB_CHARS.intersection(ref):
            matches = sorted(base.glob(ref))
            if not matches:
                raise GlobNoMatchError(f"@{ref}: no files matched")
            for m in matches:
                if m.is_file() and m not in seen:
                    seen.add(m)
                    result.append(m)
        else:
            if not p.exists():
                raise FileNotFoundAtError(f"@{ref}: file not found")
            if not p.is_file():
                raise FileNotFoundAtError(f"@{ref}: not a file")
            if p not in seen:
                seen.add(p)
                result.append(p)

    return result


_MAX_FILE_SIZE = 100 * 1024  # 100 KB


def _read_file_safe(path: Path) -> str:
    """Read file as UTF-8 text. Raises on binary, too-large, or permission errors."""
    size = path.stat().st_size
    if size > _MAX_FILE_SIZE:
        size_mb = size / (1024 * 1024)
        limit_kb = _MAX_FILE_SIZE // 1024
        raise FileTooLargeError(
            f"@{path.name}: file too large ({size_mb:.1f}MB, limit {limit_kb}KB)"
        )

    try:
        raw = path.read_bytes()
    except PermissionError:
        raise PermissionError(f"@{path.name}: permission denied")

    try:
        return raw.decode('utf-8')
    except UnicodeDecodeError:
        raise BinaryFileError(f"@{path.name}: binary file not supported")


def resolve_at_references(text: str, cwd: str) -> tuple[str, list[FileContext]]:
    """Resolve @path references in user text.

    Args:
        text: User input that may contain @path tokens
        cwd: Working directory for resolving relative paths

    Returns:
        (cleaned_text, file_contexts) — cleaned text has @path tokens removed;
        file_contexts contains path and content for each resolved file

    Raises:
        FileNotFoundAtError: A referenced path does not exist
        GlobNoMatchError: A glob pattern matched no files
        FileTooLargeError: A referenced file exceeds the size limit
        BinaryFileError: A referenced file is binary
        PermissionError: A referenced file cannot be read
    """
    refs = _scan_for_at_refs(text)
    if not refs:
        return text, []

    paths = _expand_paths(refs, cwd)

    file_contexts: list[FileContext] = []
    base = Path(cwd).resolve()
    for p in paths:
        try:
            rel = str(p.relative_to(base))
        except ValueError:
            rel = str(p)
        file_contexts.append(FileContext(path=rel, content=_read_file_safe(p)))

    # Remove @ref tokens from text, but only from segments outside backtick blocks
    # _BACKTICK_SPLIT includes captures so even indices = outside backticks, odd = inside
    parts = _BACKTICK_SPLIT.split(text)
    for i in range(0, len(parts), 2):
        for ref_path in refs:
            parts[i] = parts[i].replace(f'@{ref_path}', '', 1)
    cleaned = ' '.join(''.join(parts).split())

    return cleaned, file_contexts


def _build_context_block(files: list[FileContext]) -> str:
    """Build a labeled context block string for injection into the LLM prompt."""
    parts = ["[Files referenced by the user:]\n"]
    for f in files:
        parts.append(f"--- {f.path} ---\n{f.content}\n")
    return "\n".join(parts)


class AtFileCompleter(Completer):
    """prompt_toolkit Completer that activates on @-prefixed words.

    Strips the @ prefix, completes paths relative to cwd via pathlib,
    then re-adds @ to the completion text.
    """

    def __init__(self, cwd: str) -> None:
        self._cwd = Path(cwd)

    def get_completions(self, document: Document, complete_event):
        word = document.get_word_before_cursor(WORD=True)
        if not word.startswith("@"):
            return

        path_part = word[1:]  # strip @ prefix

        # Resolve parent directory and filename prefix
        if "/" in path_part:
            dir_part = path_part.rsplit("/", 1)[0]
            prefix = path_part.rsplit("/", 1)[1]
            parent = self._cwd / dir_part
        else:
            dir_part = ""
            parent = self._cwd
            prefix = path_part

        try:
            entries = sorted(parent.iterdir())
        except (OSError, FileNotFoundError):
            return

        for child in entries:
            if child.name.startswith(prefix):
                rel = f"{dir_part}/{child.name}" if dir_part else child.name
                yield Completion(
                    text=f"@{rel}",
                    start_position=-len(word),
                    display=rel,
                )
