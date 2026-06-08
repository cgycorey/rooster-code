"""Persistent markdown-based memory system for rooster-code.

Memory files live in ~/.rooster-code/memory/ (global) and
.rooster-code/memory/ (project). Each file has YAML frontmatter
with name, description, and type fields. MEMORY.md serves as an index.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

GLOBAL_MEMORY_DIR = Path.home() / ".rooster-code" / "memory"
PROJECT_MEMORY_DIR = Path(".rooster-code") / "memory"
MEMORY_INDEX = "MEMORY.md"
MEMORY_MAX_COUNT = 20

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_yaml_value(raw: str) -> str:
    """Strip YAML quoting and unescape a frontmatter value."""
    val = raw.strip()
    if len(val) >= 2 and val[0] == '"' and val[-1] == '"':
        # Double-quoted: unescape inner content.
        # Process \\ first (via sentinel) so that \\n (escaped backslash
        # followed by literal n) is not mistaken for \n (newline).
        # "\\\\" in a Python string literal is one backslash character,
        # so we use a raw string r"\\" (or quadruple "\\\\\\\\") to match
        # two consecutive backslashes (the YAML escape for one literal backslash).
        inner = val[1:-1]
        SENTINEL = "\x00"
        inner = inner.replace(r"\\", SENTINEL)
        inner = inner.replace('\\"', '"')
        inner = inner.replace("\\n", "\n")
        inner = inner.replace(SENTINEL, "\\")
        return inner
    if len(val) >= 2 and val[0] == "'" and val[-1] == "'":
        # Single-quoted: literal content
        return val[1:-1]
    return val


def _parse_frontmatter(content: str) -> tuple[dict[str, str], str]:
    """Parse YAML frontmatter from memory file content. Returns (fields, body)."""
    m = _FRONTMATTER_RE.match(content)
    if not m:
        return {}, content
    fields: dict[str, str] = {}
    for line in m.group(1).split("\n"):
        line = line.strip()
        if ":" in line:
            key, _, val = line.partition(":")
            fields[key.strip()] = _parse_yaml_value(val)
    body = content[m.end():].strip()
    return fields, body


def _read_file_safe(path: Path) -> str | None:
    """Read a file safely with O_NOFOLLOW. Returns content or None."""
    fd = None
    try:
        fd = os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW)
    except OSError:
        return None
    try:
        with open(fd, "r", encoding="utf-8", closefd=False) as fh:
            return fh.read()
    except OSError:
        log.exception("Could not read memory file %s", path)
        return None
    finally:
        if fd is not None:
            os.close(fd)


def _load_memory_dir(directory: Path) -> list[dict[str, str]]:
    """Load all memory files from a directory. Returns list of {name, description, content}.

    Sorted by modification time (newest first) so truncation keeps recent memories.
    """
    memories: list[dict[str, str]] = []
    if not directory.is_dir():
        return memories
    try:
        entries = sorted(
            directory.iterdir(),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for entry in entries:
            if not entry.is_file() or not entry.suffix == ".md":
                continue
            if entry.name == MEMORY_INDEX:
                continue
            text = _read_file_safe(entry)
            if text is None:
                continue
            fields, body = _parse_frontmatter(text)
            name = fields.get("name", entry.stem)
            desc = fields.get("description", "")
            memories.append({"name": name, "description": desc, "content": body})
    except OSError:
        log.exception("Could not iterate memory directory %s", directory)
    return memories


def _find_memory_file(directory: Path, name: str) -> Path | None:
    """Find a memory file by its frontmatter name.

    Tries the current hash-suffixed slug first, then falls back to scanning
    frontmatter for backward compatibility with old-style (hashless) filenames.
    """
    if not directory.is_dir():
        return None
    # Try current hashed slug first
    hashed = directory / f"{_slugify(name)}.md"
    if hashed.is_file():
        return hashed
    # Fall back: scan for matching frontmatter name (backward compat)
    try:
        for entry in directory.iterdir():
            if not entry.is_file() or not entry.suffix == ".md":
                continue
            if entry.name == MEMORY_INDEX:
                continue
            text = _read_file_safe(entry)
            if text is None:
                continue
            fields, _ = _parse_frontmatter(text)
            if fields.get("name") == name:
                return entry
    except OSError:
        log.exception("Could not scan memory directory %s", directory)
    return None


def _slugify(name: str) -> str:
    """Convert a memory name to a safe filename slug with collision-resistant hash suffix."""
    base = re.sub(r"[^a-z0-9_-]", "", name.lower().replace(" ", "-")) or "memory"
    h = hashlib.md5(name.encode()).hexdigest()[:4]
    return f"{base}-{h}"


def load_memories() -> list[dict[str, str]]:
    """Load all memory entries with metadata. Project memories come first
    so truncation to MEMORY_MAX_COUNT favors project-relevant entries."""
    return _load_memory_dir(PROJECT_MEMORY_DIR) + _load_memory_dir(GLOBAL_MEMORY_DIR)


def build_memory_prompt_section() -> str:
    """Build the memory prompt section from all global and project memory files.

    User-supplied content is wrapped in memory tags to isolate it from
    the system prompt and prevent prompt injection via saved memory content.
    Capped at MEMORY_MAX_COUNT to prevent context overflow; project memories
    come first, and within each directory newest (by mtime) are kept first.
    """
    all_memories = load_memories()
    if not all_memories:
        return ""
    total_count = len(all_memories)
    truncated = total_count > MEMORY_MAX_COUNT
    if truncated:
        all_memories = all_memories[:MEMORY_MAX_COUNT]
    lines: list[str] = ["# Saved Memories"]
    if truncated:
        lines.append(f"_(showing {MEMORY_MAX_COUNT} of {total_count} memories)_")
    for mem in all_memories:
        lines.append(f"\n## {mem['name']}")
        if mem["description"]:
            lines.append(f"_{mem['description']}_")
        lines.append("")
        lines.append("<memory>")
        lines.append(mem["content"].replace("</memory>", "<\\/memory>"))
        lines.append("</memory>")
    return "\n".join(lines)


def _escape_yaml_value(value: str) -> str:
    """Escape a string for safe use as a YAML value. If it contains newlines
    or special characters, use a double-quoted string with escaping.
    Block scalars are intentionally avoided because _parse_frontmatter
    cannot parse them back."""
    if "\n" in value:
        # Use double-quoted with \n escape so the reader roundtrips
        escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        return '"' + escaped + '"'
    if any(c in value for c in (':', '#', '"', "'", '&', '*', '!', '>', '|', '%', '@', '`', '{', '}', '[', ']')):
        return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'
    return value


def save_memory(name: str, content: str, description: str = "", *, global_scope: bool = False) -> Path:
    """Save a memory file. Creates the directory if needed. Returns the file path.

    Uses atomic write via temp file + os.replace to prevent TOCTOU races
    and partial writes. Frontmatter values are escaped to prevent YAML injection.
    Cleans up any old-style (pre-hash) file for the same memory name so
    upgrades don't produce duplicate entries.
    """
    target_dir = GLOBAL_MEMORY_DIR if global_scope else PROJECT_MEMORY_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    file_path = target_dir / f"{_slugify(name)}.md"

    # Find any pre-existing file (possibly old-style hashless) to clean up
    old_file = _find_memory_file(target_dir, name)
    if old_file is not None and old_file != file_path:
        # An old-style file exists — we'll replace it, then remove the old one
        pass

    frontmatter = (
        f"---\n"
        f"name: {_escape_yaml_value(name)}\n"
        f"description: {_escape_yaml_value(description)}\n"
        f"type: memory\n"
        f"---\n\n"
    )

    # Atomic write: write to temp file then rename
    fd, tmp_path = tempfile.mkstemp(dir=str(target_dir), suffix=".tmp")
    try:
        with open(fd, "w", encoding="utf-8", closefd=False) as fh:
            fh.write(frontmatter + content)
        os.replace(tmp_path, str(file_path))
    finally:
        os.close(fd)
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Remove any old-style duplicate so _load_memory_dir doesn't double-load
    if old_file is not None and old_file != file_path:
        try:
            old_file.unlink()
        except OSError:
            pass

    return file_path


def delete_memory(name: str) -> Path | None:
    """Delete a memory file by name. Returns the path if deleted, None if not found.

    Uses _find_memory_file for backward-compatible filename resolution
    (old-style hashless slugs and new hash-suffixed slugs).
    """
    for d in (PROJECT_MEMORY_DIR, GLOBAL_MEMORY_DIR):
        fp = _find_memory_file(d, name)
        if fp is None:
            continue
        try:
            fp.unlink()
            return fp
        except OSError:
            log.exception("Could not delete memory file %s", fp)
            return None
    return None
