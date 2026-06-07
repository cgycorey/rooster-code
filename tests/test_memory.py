"""Tests for the rooster-code memory system."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from rooster_code.memory import (
    _parse_frontmatter,
    build_memory_prompt_section,
    delete_memory,
    load_memories,
    save_memory,
    MEMORY_MAX_COUNT,
)


def test_parse_frontmatter() -> None:
    text = "---\nname: test\ndescription: a test\ntype: memory\n---\n\nThis is the body."
    fields, body = _parse_frontmatter(text)
    assert fields["name"] == "test"
    assert fields["description"] == "a test"
    assert fields["type"] == "memory"
    assert body == "This is the body."


def test_parse_frontmatter_no_frontmatter() -> None:
    fields, body = _parse_frontmatter("Just plain text")
    assert fields == {}
    assert body == "Just plain text"


def test_save_and_load_memory() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        with patch("rooster_code.memory.PROJECT_MEMORY_DIR", Path(tmp)):
            assert load_memories() == []
            file_path = save_memory("Test Memory", "Hello world", "test desc")
            assert file_path.exists()
            content = file_path.read_text()
            assert "Test Memory" in content
            assert "Hello world" in content

            memories = load_memories()
            assert len(memories) == 1
            assert memories[0]["name"] == "Test Memory"
            assert memories[0]["content"] == "Hello world"


def test_build_memory_prompt_section_empty() -> None:
    with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
        with (
            patch("rooster_code.memory.GLOBAL_MEMORY_DIR", Path(tmp1)),
            patch("rooster_code.memory.PROJECT_MEMORY_DIR", Path(tmp2)),
        ):
            assert build_memory_prompt_section() == ""


def test_build_memory_prompt_section_with_memories() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "test.md").write_text(
            "---\nname: Test\n---\n\nImportant content."
        )
        with patch("rooster_code.memory.PROJECT_MEMORY_DIR", Path(tmp)), patch(
            "rooster_code.memory.GLOBAL_MEMORY_DIR", Path(tmp)
        ):
            section = build_memory_prompt_section()
            assert "# Saved Memories" in section
            assert "Test" in section
            assert "Important content" in section


def test_build_memory_prompt_section_skips_memory_index() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "MEMORY.md").write_text("index")
        Path(tmp, "real.md").write_text("---\nname: Real\n---\n\nReal content.")
        with patch("rooster_code.memory.PROJECT_MEMORY_DIR", Path(tmp)), patch(
            "rooster_code.memory.GLOBAL_MEMORY_DIR", Path(tmp)
        ):
            section = build_memory_prompt_section()
            assert "Real content" in section
            assert "MEMORY" not in section


def test_delete_memory() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        with patch("rooster_code.memory.PROJECT_MEMORY_DIR", Path(tmp)):
            save_memory("to-delete", "content")
            assert len(load_memories()) == 1
            result = delete_memory("to-delete")
            assert result is not None
            assert len(load_memories()) == 0


def test_delete_memory_not_found() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        with patch("rooster_code.memory.PROJECT_MEMORY_DIR", Path(tmp)):
            result = delete_memory("nonexistent")
            assert result is None


def test_save_memory_duplicate_overwrites() -> None:
    """save_memory overwrites existing files — last write wins."""
    with tempfile.TemporaryDirectory() as tmp:
        with patch("rooster_code.memory.PROJECT_MEMORY_DIR", Path(tmp)):
            save_memory("dup", "v1")
            save_memory("dup", "v2")
            memories = load_memories()
            assert len(memories) == 1
            assert memories[0]["content"] == "v2"


def test_build_memory_prompt_section_truncated() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        with patch("rooster_code.memory.PROJECT_MEMORY_DIR", Path(tmp)), patch(
            "rooster_code.memory.GLOBAL_MEMORY_DIR", Path(tmp)
        ):
            # Create MEMORY_MAX_COUNT + 5 memories
            for i in range(MEMORY_MAX_COUNT + 5):
                fd, fp = tempfile.mkstemp(dir=tmp, suffix=".md")
                os.close(fd)
                Path(fp).write_text(f"---\nname: Mem{i}\n---\n\nContent {i}.")
            section = build_memory_prompt_section()
            assert "showing" in section
            # Only MEMORY_MAX_COUNT sections should appear
            name_count = section.count("## Mem")
            assert name_count == MEMORY_MAX_COUNT


def test_slugify_collision_resistance() -> None:
    """Different memory names must produce different slugs."""
    from rooster_code.memory import _slugify

    slug1 = _slugify("My Project")
    slug2 = _slugify("my-project")
    slug3 = _slugify("my_project")
    assert slug1 != slug2, f"Collision: {slug1!r} == {slug2!r}"
    assert slug1 != slug3, f"Collision: {slug1!r} == {slug3!r}"
    assert slug2 != slug3, f"Collision: {slug2!r} == {slug3!r}"


def test_memory_prompt_section_includes_memory_tags() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "tagged.md").write_text("---\nname: Tagged\n---\n\nSafe content.")
        with patch("rooster_code.memory.PROJECT_MEMORY_DIR", Path(tmp)), patch(
            "rooster_code.memory.GLOBAL_MEMORY_DIR", Path(tmp)
        ):
            section = build_memory_prompt_section()
            assert "<memory>" in section
            assert "</memory>" in section
