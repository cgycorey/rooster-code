"""Persistent markdown-based memory system for rooster-code.

Memory files live in ~/.rooster-code/memory/ (global) and
.rooster-code/memory/ (project). Each file has YAML frontmatter
with name, description, and type fields. MEMORY.md serves as an index.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

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
    try:
        fd = os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW)
    except OSError:
        return None
    try:
        with open(fd, "r", encoding="utf-8", closefd=False) as fh:
            return fh.read()
    except OSError:
        return None
    finally:
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
        pass
    return memories


def _slugify(name: str) -> str:
    """Convert a memory name to a safe filename slug."""
    return re.sub(r"[^a-z0-9_-]", "", name.lower().replace(" ", "-")) or "memory"


def load_memories() -> list[dict[str, str]]:
    """Load all memory entries with metadata."""
    return _load_memory_dir(GLOBAL_MEMORY_DIR) + _load_memory_dir(PROJECT_MEMORY_DIR)


def build_memory_prompt_section() -> str:
    """Build the memory prompt section from all global and project memory files."""
    all_memories = load_memories()
    if not all_memories:
        return ""
    lines: list[str] = ["# Saved Memories"]
    for mem in all_memories:
        lines.append(f"\n## {mem['name']}")
        if mem["description"]:
            lines.append(f"_{mem['description']}_")
        lines.append("")
        lines.append(mem["content"])
    return "\n".join(lines)


def save_memory(name: str, content: str, description: str = "", *, global_scope: bool = False) -> Path:
    """Save a memory file. Creates the directory if needed. Returns the file path."""
    target_dir = GLOBAL_MEMORY_DIR if global_scope else PROJECT_MEMORY_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    file_path = target_dir / f"{_slugify(name)}.md"
    frontmatter = f"---\nname: {name}\ndescription: {description}\ntype: memory\n---\n\n"
    file_path.write_text(frontmatter + content, encoding="utf-8")
    return file_path


def delete_memory(name: str) -> Path | None:
    """Delete a memory file by name. Returns the path if deleted, None if not found."""
    slug = _slugify(name)
    for d in (PROJECT_MEMORY_DIR, GLOBAL_MEMORY_DIR):
        fp = d / f"{slug}.md"
        if fp.exists():
            try:
                fp.unlink()
                return fp
            except OSError:
                pass
    return None
