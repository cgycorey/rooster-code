"""Tests for the rooster-code memory system."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from rooster_code.memory import (
    _parse_frontmatter,
    build_memory_prompt_section,
    load_memories,
    save_memory,
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
