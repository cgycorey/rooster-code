"""Test actual CLI command parsing for /handoff with path arguments.

These tests verify how parse_chat_command tokenizes /handoff inputs
and how the CLI resolves the resulting path against config.cwd.
"""

from pathlib import Path

from rooster_code.chat import parse_chat_command


def _resolve_path(raw_input: str, cwd: Path | None = None) -> str:
    """Simulate the CLI's path resolution for /handoff."""
    command = parse_chat_command(raw_input)
    if command.name != "handoff":
        raise ValueError(f"Not a handoff command: {raw_input!r}")
    base = cwd or Path(".")  # type: ignore[arg-type]
    if command.args:
        return str(base / " ".join(command.args))
    return str(base / ".handoff")


class TestHandoffCommandParsing:
    """Cover the happy-path and edge-case inputs for /handoff path resolution."""

    def test_no_args_uses_default(self, tmp_path: Path) -> None:
        assert _resolve_path("/handoff", tmp_path) == str(tmp_path / ".handoff")

    def test_single_word_arg(self, tmp_path: Path) -> None:
        assert _resolve_path("/handoff custom.handoff", tmp_path) == str(
            tmp_path / "custom.handoff"
        )

    def test_quoted_path_with_spaces(self, tmp_path: Path) -> None:
        assert _resolve_path('/handoff "my handoff.md"', tmp_path) == str(
            tmp_path / "my handoff.md"
        )

    def test_subdirectory_relative_path(self, tmp_path: Path) -> None:
        assert _resolve_path("/handoff docs/session-handoff.md", tmp_path) == str(
            tmp_path / "docs" / "session-handoff.md"
        )

    def test_absolute_path_unchanged(self, tmp_path: Path) -> None:
        abs_path = "/tmp/absolute.handoff"
        result = _resolve_path(f"/handoff {abs_path}", tmp_path)
        # Path(base / absolute) returns the absolute path unchanged
        assert result == abs_path

    def test_parent_relative_path(self, tmp_path: Path) -> None:
        assert _resolve_path("/handoff ../parent.handoff", tmp_path) == str(
            tmp_path / ".." / "parent.handoff"
        )

    def test_path_with_spaces_no_quotes(self, tmp_path: Path) -> None:
        """Without quotes, shlex splits on spaces.
        The CLI joins args with space, so the space is preserved in the filename."""
        assert _resolve_path("/handoff my file.handoff", tmp_path) == str(
            tmp_path / "my file.handoff"
        )

    def test_unquoted_path_with_spaces(self, tmp_path: Path) -> None:
        """Unquoted spaces produce multiple args that get joined back.
        '/handoff my project/handoff.md' → args=['my', 'project/handoff.md']
        → joined as 'my project/handoff.md' → resolved against cwd."""
        assert _resolve_path("/handoff my project/handoff.md", tmp_path) == str(
            tmp_path / "my project/handoff.md"
        )

    def test_dotted_filename(self, tmp_path: Path) -> None:
        assert _resolve_path("/handoff .handoff", tmp_path) == str(
            tmp_path / ".handoff"
        )

    def test_deep_nested_path(self, tmp_path: Path) -> None:
        assert _resolve_path("/handoff a/b/c/.handoff", tmp_path) == str(
            tmp_path / "a" / "b" / "c" / ".handoff"
        )

    def test_no_cwd_falls_back_to_dot(self) -> None:
        """Without cwd, the base is Path('.') which resolves to process CWD."""
        result = _resolve_path("/handoff custom.handoff")
        assert result == str(Path(".") / "custom.handoff")

    def test_single_quotes_work(self, tmp_path: Path) -> None:
        """shlex handles single quotes the same as double quotes."""
        assert _resolve_path("/handoff 'my handoff.md'", tmp_path) == str(
            tmp_path / "my handoff.md"
        )

    def test_escaped_space(self, tmp_path: Path) -> None:
        """shlex handles backslash-escaped spaces."""
        assert _resolve_path("/handoff my\\ handoff.md", tmp_path) == str(
            tmp_path / "my handoff.md"
        )
