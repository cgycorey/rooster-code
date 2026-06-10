"""@ file reference resolution — scan user text for @path tokens and resolve to file contents."""

from __future__ import annotations

import os
import re
from pathlib import Path
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

# Matches inline `...` and fenced ```...``` blocks (non-greedy)
_BACKTICK_BLOCK = re.compile(r'```[\s\S]*?```|`[^`\n]+`')
# Same pattern but capturing for re.split to include matched blocks in result
_BACKTICK_SPLIT = re.compile(r'(```[\s\S]*?```|`[^`\n]+`)')

_AT_REF = re.compile(r'(?<!\w)@(\S+)')

# Sentence punctuation that commonly trails @ references
_TRAILING_PUNCTUATION = re.compile(r'[,;:!.…]+$')


def _scan_for_at_refs(text: str) -> list[str]:
    """Find @path references in text, ignoring any inside backtick blocks."""
    safe = _BACKTICK_BLOCK.sub('', text)
    tokens = [m.group(1) for m in _AT_REF.finditer(safe)]
    return [_TRAILING_PUNCTUATION.sub('', t) for t in tokens]


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
            glob_ref = ref
            if Path(ref).is_absolute():
                glob_ref = os.path.relpath(ref, base)
                if glob_ref == ".." or glob_ref.startswith(f"..{os.sep}") or os.path.isabs(glob_ref):
                    raise FileNotFoundAtError(
                        f"@{ref}: path outside working directory"
                    )
            matches = sorted(base.glob(glob_ref))
            matches = [m for m in matches
                       if m.resolve(strict=False).is_relative_to(base)]
            if not matches:
                raise GlobNoMatchError(f"@{ref}: no files matched")
            if len(matches) > _MAX_GLOB_FILES:
                raise FileTooLargeError(
                    f"@{ref}: too many files matched "
                    f"({len(matches)}, limit {_MAX_GLOB_FILES})"
                )
            for m in matches:
                resolved = m.resolve(strict=False)
                if resolved.is_file() and resolved not in seen:
                    seen.add(resolved)
                    result.append(resolved)
        else:
            resolved = p.resolve(strict=False)
            if not resolved.is_relative_to(base):
                raise FileNotFoundAtError(
                    f"@{ref}: path outside working directory"
                )
            if not resolved.exists():
                raise FileNotFoundAtError(f"@{ref}: file not found")
            if not resolved.is_file():
                raise FileNotFoundAtError(f"@{ref}: not a file")
            if resolved not in seen:
                seen.add(resolved)
                result.append(resolved)

    return result


_MAX_FILE_SIZE = 512 * 1024  # 512 KB
_MAX_GLOB_FILES = 200


def _read_file_safe(path: Path) -> str:
    """Read file as UTF-8 text. Raises on binary, too-large, or permission errors."""
    try:
        fd = os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW)
    except PermissionError:
        raise PermissionError(f"@{path.name}: permission denied")
    except OSError as exc:
        raise OSError(f"@{path.name}: {exc}")

    try:
        stat = os.fstat(fd)
        size = stat.st_size
        if size > _MAX_FILE_SIZE:
            size_mb = size / (1024 * 1024)
            limit_kb = _MAX_FILE_SIZE // 1024
            raise FileTooLargeError(
                f"@{path.name}: file too large ({size_mb:.1f}MB, limit {limit_kb}KB)"
            )

        chunks: list[bytes] = []
        remaining = size
        while remaining > 0:
            chunk = os.read(fd, min(remaining, 65536))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
    finally:
        os.close(fd)

    try:
        return raw.decode("utf-8")
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
    cleaned = ''.join(parts).strip()
    cleaned = re.sub(r' {2,}', ' ', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

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
