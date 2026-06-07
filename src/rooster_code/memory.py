"""Persistent markdown-based memory system for rooster-code.

Memory files live in ~/.rooster-code/memory/ (global) and
.rooster-code/memory/ (project). Each file has YAML frontmatter
with name, description, and type fields. MEMORY.md serves as an index.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

GLOBAL_MEMORY_DIR = Path.home() / ".rooster-code" / "memory"
PROJECT_MEMORY_DIR = Path(".rooster-code") / "memory"
MEMORY_INDEX = "MEMORY.md"

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


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
            fields[key.strip()] = val.strip()
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
    """Load all memory files from a directory. Returns list of {name, description, content}."""
    memories: list[dict[str, str]] = []
    if not directory.is_dir():
        return memories
    try:
        for entry in sorted(directory.iterdir()):
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


def _slugify(name: str) -> str:
    """Convert a memory name to a safe filename slug."""
    return re.sub(r"[^a-z0-9_-]", "", name.lower().replace(" ", "-")) or "memory"


def load_memories() -> list[dict[str, str]]:
    """Load all memory entries with metadata."""
    return _load_memory_dir(GLOBAL_MEMORY_DIR) + _load_memory_dir(PROJECT_MEMORY_DIR)


def build_memory_prompt_section() -> str:
    """Build the memory prompt section from all global and project memory files.

    User-supplied content is wrapped in memory tags to isolate it from
    the system prompt and prevent prompt injection via saved memory content.
    """
    all_memories = load_memories()
    if not all_memories:
        return ""
    lines: list[str] = ["# Saved Memories"]
    for mem in all_memories:
        lines.append(f"\n## {mem['name']}")
        if mem["description"]:
            lines.append(f"_{mem['description']}_")
        lines.append("")
        lines.append("<memory>")
        lines.append(mem["content"])
        lines.append("</memory>")
    return "\n".join(lines)


def _escape_yaml_value(value: str) -> str:
    """Escape a string for safe use as a YAML value. If it contains newlines
    or special characters, use a quoted string with escaping."""
    if "\n" in value:
        # Use a literal block scalar for multi-line values
        indented = "\n".join("  " + line for line in value.split("\n"))
        return "|\n" + indented
    if any(c in value for c in (':', '#', '"', "'", '&', '*', '!', '>', '|', '%', '@', '`', '{', '}', '[', ']')):
        return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'
    return value


def save_memory(name: str, content: str, description: str = "", *, global_scope: bool = False) -> Path:
    """Save a memory file. Creates the directory if needed. Returns the file path.

    Uses atomic write via temp file + os.replace to prevent TOCTOU races
    and partial writes. Frontmatter values are escaped to prevent YAML injection.
    """
    target_dir = GLOBAL_MEMORY_DIR if global_scope else PROJECT_MEMORY_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    file_path = target_dir / f"{_slugify(name)}.md"

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

    return file_path


def delete_memory(name: str) -> Path | None:
    """Delete a memory file by name. Returns the path if deleted, None if not found."""
    slug = _slugify(name)
    for d in (PROJECT_MEMORY_DIR, GLOBAL_MEMORY_DIR):
        fp = d / f"{slug}.md"
        try:
            fp.unlink()
            return fp
        except FileNotFoundError:
            continue
        except OSError:
            log.exception("Could not delete memory file %s", fp)
            return None
    return None
