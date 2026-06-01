"""Tests for rooster_code.file_context — @ reference resolution and AtFileCompleter."""

import os
from pathlib import Path

import pytest
from prompt_toolkit.document import Document

from rooster_code.file_context import (
    _scan_for_at_refs,
    _expand_paths,
    _read_file_safe,
    resolve_at_references,
    _build_context_block,
    FileContext,
    FileNotFoundAtError,
    GlobNoMatchError,
    BinaryFileError,
    FileTooLargeError,
    AtFileCompleter,
)


class TestScanForAtRefs:
    def test_single_at_ref(self):
        refs = _scan_for_at_refs("check @src/foo.py please")
        assert refs == ["src/foo.py"]

    def test_multiple_at_refs(self):
        refs = _scan_for_at_refs("@src/a.py and @src/b.py")
        assert refs == ["src/a.py", "src/b.py"]

    def test_no_at_ref(self):
        refs = _scan_for_at_refs("no references here")
        assert refs == []

    def test_at_inside_inline_backtick_ignored(self):
        refs = _scan_for_at_refs("see `@src/foo.py` here and @src/bar.py outside")
        assert refs == ["src/bar.py"]

    def test_at_inside_fenced_backtick_ignored(self):
        refs = _scan_for_at_refs(
            "before\n```\n@src/foo.py\n```\nafter @src/bar.py"
        )
        assert refs == ["src/bar.py"]

    def test_at_symbol_only_ignored(self):
        refs = _scan_for_at_refs("email me at @ but not a file")
        assert refs == []

    def test_glob_pattern(self):
        refs = _scan_for_at_refs("check @src/**/*.py")
        assert refs == ["src/**/*.py"]

    def test_trailing_comma_stripped(self):
        refs = _scan_for_at_refs("check @foo.py, and more")
        assert refs == ["foo.py"]

    def test_trailing_period_stripped(self):
        refs = _scan_for_at_refs("read @bar.py.")
        assert refs == ["bar.py"]

    def test_trailing_semicolon_stripped(self):
        refs = _scan_for_at_refs("@baz.py; next")
        assert refs == ["baz.py"]

    def test_trailing_multiple_punctuation(self):
        refs = _scan_for_at_refs("@file.py,... check")
        assert refs == ["file.py"]

    def test_glob_question_mark_not_stripped(self):
        refs = _scan_for_at_refs("check @file?.py")
        assert refs == ["file?.py"]


class TestExpandPaths:
    def test_single_file(self, tmp_path):
        (tmp_path / "foo.py").write_text("hello")
        paths = _expand_paths(["foo.py"], str(tmp_path))
        assert paths == [tmp_path / "foo.py"]

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundAtError, match="noexist.py"):
            _expand_paths(["noexist.py"], str(tmp_path))

    def test_glob_expansion(self, tmp_path):
        (tmp_path / "a.py").write_text("a")
        (tmp_path / "b.py").write_text("b")
        (tmp_path / "c.txt").write_text("c")
        paths = _expand_paths(["*.py"], str(tmp_path))
        assert sorted(paths) == sorted([tmp_path / "a.py", tmp_path / "b.py"])

    def test_glob_no_match(self, tmp_path):
        with pytest.raises(GlobNoMatchError, match="no files matched"):
            _expand_paths(["*.rs"], str(tmp_path))

    def test_duplicate_dedup(self, tmp_path):
        (tmp_path / "foo.py").write_text("hello")
        paths = _expand_paths(["foo.py", "foo.py"], str(tmp_path))
        assert paths == [tmp_path / "foo.py"]

    def test_absolute_path(self, tmp_path):
        p = tmp_path / "abs.py"
        p.write_text("abs")
        paths = _expand_paths([str(p)], str(tmp_path))
        assert paths == [p]

    def test_multiple_refs_mixed(self, tmp_path):
        (tmp_path / "a.py").write_text("a")
        (tmp_path / "b.py").write_text("b")
        paths = _expand_paths(["a.py", "b.py"], str(tmp_path))
        assert sorted(paths) == sorted([tmp_path / "a.py", tmp_path / "b.py"])

    def test_path_traversal_rejected(self, tmp_path):
        with pytest.raises(FileNotFoundAtError, match="outside working directory"):
            _expand_paths(["../outside.py"], str(tmp_path))

    def test_absolute_outside_rejected(self, tmp_path):
        with pytest.raises(FileNotFoundAtError, match="outside working directory"):
            _expand_paths(["/etc/passwd"], str(tmp_path))

    def test_absolute_inside_accepted(self, tmp_path):
        (tmp_path / "ok.py").write_text("ok")
        paths = _expand_paths([str(tmp_path / "ok.py")], str(tmp_path))
        assert paths == [tmp_path / "ok.py"]

    def test_symlink_inside_accepted(self, tmp_path):
        (tmp_path / "real.py").write_text("real")
        os.symlink(str(tmp_path / "real.py"), str(tmp_path / "link.py"))
        paths = _expand_paths(["link.py"], str(tmp_path))
        assert len(paths) == 1
        assert paths[0] == (tmp_path / "real.py")

    def test_symlink_outside_rejected(self, tmp_path):
        os.symlink("/etc/passwd", str(tmp_path / "escape_link.py"))
        with pytest.raises(FileNotFoundAtError, match="outside working directory"):
            _expand_paths(["escape_link.py"], str(tmp_path))

    def test_glob_limit_exceeded(self, tmp_path, monkeypatch):
        monkeypatch.setattr("rooster_code.file_context._MAX_GLOB_FILES", 2)
        for i in range(10):
            (tmp_path / f"file_{i}.py").write_text("x")
        with pytest.raises(FileTooLargeError, match="too many files matched"):
            _expand_paths(["*.py"], str(tmp_path))

    def test_glob_limit_not_exceeded(self, tmp_path, monkeypatch):
        monkeypatch.setattr("rooster_code.file_context._MAX_GLOB_FILES", 10)
        for i in range(3):
            (tmp_path / f"file_{i}.py").write_text("x")
        paths = _expand_paths(["*.py"], str(tmp_path))
        assert len(paths) == 3


class TestReadFileSafe:
    def test_read_text_file(self, tmp_path):
        p = tmp_path / "hello.py"
        p.write_text("print('hello')")
        content = _read_file_safe(p)
        assert content == "print('hello')"

    def test_binary_file_raises(self, tmp_path):
        p = tmp_path / "data.bin"
        p.write_bytes(b'\x00\x01\x02\xff\xfe\xfd')
        with pytest.raises(BinaryFileError, match="data.bin"):
            _read_file_safe(p)

    def test_file_too_large_raises(self, tmp_path, monkeypatch):
        p = tmp_path / "big.log"
        p.write_text("x" * 200_000)
        monkeypatch.setattr(
            "rooster_code.file_context._MAX_FILE_SIZE", 100_000
        )
        with pytest.raises(FileTooLargeError, match="big.log"):
            _read_file_safe(p)

    def test_file_under_limit_ok(self, tmp_path):
        p = tmp_path / "ok.log"
        p.write_text("x" * 400_000)
        content = _read_file_safe(p)
        assert len(content) == 400_000

    def test_permission_error(self, tmp_path):
        p = tmp_path / "nope.txt"
        p.write_text("secret")
        p.chmod(0o000)
        try:
            with pytest.raises(PermissionError):
                _read_file_safe(p)
        finally:
            p.chmod(0o644)

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.txt"
        p.write_text("")
        content = _read_file_safe(p)
        assert content == ""


class TestResolveAtReferences:
    def test_single_file_returns_content(self, tmp_path):
        (tmp_path / "foo.py").write_text("x = 1")
        cleaned, files = resolve_at_references("look at @foo.py", str(tmp_path))
        assert cleaned == "look at"
        assert len(files) == 1
        assert files[0].path == "foo.py"
        assert files[0].content == "x = 1"

    def test_multiple_files(self, tmp_path):
        (tmp_path / "a.py").write_text("a")
        (tmp_path / "b.py").write_text("b")
        cleaned, files = resolve_at_references("@a.py and @b.py", str(tmp_path))
        assert cleaned == "and"
        assert len(files) == 2
        assert {f.path for f in files} == {"a.py", "b.py"}
        assert {f.content for f in files} == {"a", "b"}

    def test_no_at_refs_returns_unchanged(self, tmp_path):
        cleaned, files = resolve_at_references("hello world", str(tmp_path))
        assert cleaned == "hello world"
        assert files == []

    def test_glob_expansion(self, tmp_path):
        (tmp_path / "x.py").write_text("x")
        (tmp_path / "y.py").write_text("y")
        cleaned, files = resolve_at_references("@*.py look", str(tmp_path))
        assert cleaned == "look"
        assert len(files) == 2

    def test_file_not_found_error(self, tmp_path):
        with pytest.raises(FileNotFoundAtError, match="nope.py"):
            resolve_at_references("@nope.py check", str(tmp_path))

    def test_glob_no_match_error(self, tmp_path):
        with pytest.raises(GlobNoMatchError, match="no files matched"):
            resolve_at_references("@*.rs check", str(tmp_path))

    def test_at_inside_backtick_ignored(self, tmp_path):
        (tmp_path / "bar.py").write_text("bar")
        cleaned, files = resolve_at_references(
            "`@bar.py` but @bar.py", str(tmp_path)
        )
        # The @bar.py inside backticks stays intact; the outside one was removed
        assert "`@bar.py`" in cleaned
        assert cleaned.count("@bar.py") == 1
        assert len(files) == 1

    def test_message_is_only_at_ref(self, tmp_path):
        (tmp_path / "foo.py").write_text("x = 1")
        cleaned, files = resolve_at_references("@foo.py", str(tmp_path))
        assert cleaned == ""
        assert len(files) == 1

    def test_trailing_comma_resolves(self, tmp_path):
        (tmp_path / "foo.py").write_text("x = 1")
        cleaned, files = resolve_at_references("@foo.py, check", str(tmp_path))
        assert len(files) == 1
        assert files[0].content == "x = 1"

    def test_trailing_period_resolves(self, tmp_path):
        (tmp_path / "bar.py").write_text("bar")
        cleaned, files = resolve_at_references("see @bar.py.", str(tmp_path))
        assert len(files) == 1
        assert files[0].content == "bar"

    def test_path_traversal_rejected_integration(self, tmp_path):
        with pytest.raises(FileNotFoundAtError, match="outside"):
            resolve_at_references("@../secret.txt check", str(tmp_path))

    def test_newlines_preserved(self, tmp_path):
        (tmp_path / "foo.py").write_text("x = 1")
        cleaned, files = resolve_at_references(
            "line one\n\n@foo.py\n\nline two", str(tmp_path)
        )
        assert "line one" in cleaned
        assert "line two" in cleaned
        assert "\n" in cleaned


class TestBuildContextBlock:
    def test_single_file(self):
        block = _build_context_block([FileContext("foo.py", "x = 1")])
        assert "--- foo.py ---" in block
        assert "x = 1" in block
        assert "[Files referenced by the user:]" in block

    def test_multiple_files(self):
        files = [
            FileContext("a.py", "aaa"),
            FileContext("b.py", "bbb"),
        ]
        block = _build_context_block(files)
        assert "--- a.py ---" in block
        assert "--- b.py ---" in block
        assert block.index("--- a.py ---") < block.index("--- b.py ---")


class TestAtFileCompleter:
    def test_returns_completions_for_at_prefix(self, tmp_path):
        (tmp_path / "foo.py").write_text("x")
        (tmp_path / "bar.py").write_text("y")
        completer = AtFileCompleter(cwd=str(tmp_path))
        doc = Document("@f", 2)
        completions = list(completer.get_completions(doc, None))
        texts = {c.text for c in completions}
        assert "@foo.py" in texts

    def test_no_completions_without_at_prefix(self, tmp_path):
        (tmp_path / "foo.py").write_text("x")
        completer = AtFileCompleter(cwd=str(tmp_path))
        doc = Document("f", 1)
        completions = list(completer.get_completions(doc, None))
        assert completions == []

    def test_completions_for_directory(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "a.py").write_text("a")
        completer = AtFileCompleter(cwd=str(tmp_path))
        doc = Document("@sub/", 5)
        completions = list(completer.get_completions(doc, None))
        texts = {c.text for c in completions}
        assert "@sub/a.py" in texts

    def test_empty_at_prefix(self, tmp_path):
        (tmp_path / "readme.md").write_text("doc")
        completer = AtFileCompleter(cwd=str(tmp_path))
        doc = Document("@", 1)
        completions = list(completer.get_completions(doc, None))
        texts = {c.text for c in completions}
        assert "@readme.md" in texts
