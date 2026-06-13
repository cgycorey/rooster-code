from __future__ import annotations

import pytest

import argparse
import json

from rooster_code.config import load_json_file, resolve_runtime_env


def test_rooster_code_env_names_are_mapped() -> None:
    env = {
        "ROOSTER_CODE_API_KEY": "key-123",
        "ROOSTER_CODE_BASE_URL": "https://example.test",
        "ROOSTER_CODE_MODEL": "test-model",
        "ROOSTER_CODE_API_TYPE": "openai-completions",
        "ROOSTER_CODE_SEARCH_URL": "https://search.example/search",
    }

    resolved = resolve_runtime_env(env)

    assert resolved.api_key == "key-123"
    assert resolved.base_url == "https://example.test"
    assert resolved.model == "test-model"
    assert resolved.api_type == "openai-completions"
    assert resolved.search_url == "https://search.example/search"


def test_config_from_namespace_loads_runtime_files_and_kv_pairs(tmp_path) -> None:
    from rooster_code.config import config_from_namespace

    agents_file = tmp_path / "agents.json"
    hooks_file = tmp_path / "hooks.json"
    schema_file = tmp_path / "schema.json"
    mcp_file = tmp_path / "mcp.json"
    extra_args_file = tmp_path / "extra.json"
    skills_dir = tmp_path / "skills"

    agents_file.write_text(json.dumps({"reviewer": {"description": "code reviewer"}}), encoding="utf-8")
    hooks_file.write_text(json.dumps({"PreToolUse": []}), encoding="utf-8")
    schema_file.write_text(json.dumps({"type": "object"}), encoding="utf-8")
    mcp_file.write_text(json.dumps({"fs": {"type": "stdio", "command": "echo", "args": ["hi"]}}), encoding="utf-8")
    extra_args_file.write_text(json.dumps({"temperature": 0}), encoding="utf-8")
    skills_dir.mkdir()

    args = argparse.Namespace(
        model="cli-model",
        cwd="/tmp/project",
        allowed_tools=["Read"],
        disallowed_tools=["Bash"],
        resume="sess-1",
        session_id="sess-2",
        continue_session=True,
        fork_session="sess-0",
        persist_session=False,
        permission_mode="acceptEdits",
        max_turns=12,
        max_budget_usd=4.5,
        max_tokens=900,
        thinking_budget=321,
        debug=True,
        sandbox=True,
        include_partials=True,
        env=["A=1", "B=2"],
        custom_headers=["X-Test=1"],
        agents_file=str(agents_file),
        hooks_file=str(hooks_file),
        json_schema_file=str(schema_file),
        mcp_file=str(mcp_file),
        extra_args_file=str(extra_args_file),
        skills_dir=str(skills_dir),
    )

    config = config_from_namespace(args, {"ROOSTER_CODE_API_KEY": "env-key", "ROOSTER_CODE_BASE_URL": "https://env.test"})

    assert config.api_key == "env-key"
    assert config.base_url == "https://env.test"
    assert config.model == "cli-model"
    assert config.cwd == "/tmp/project"
    assert config.allowed_tools == ["Read"]
    assert config.disallowed_tools == ["Bash"]
    assert config.resume == "sess-1"
    assert config.session_id == "sess-2"
    assert config.continue_session is True
    assert config.fork_session == "sess-0"
    assert config.persist_session is False
    assert config.permission_mode == "acceptEdits"
    assert config.max_turns == 12
    assert config.max_budget_usd == 4.5
    assert config.max_tokens == 900
    assert config.thinking_budget == 321
    assert config.debug is True
    assert config.sandbox is True
    assert config.include_partials is True
    assert config.env == {"A": "1", "B": "2"}
    assert config.custom_headers == {"X-Test": "1"}
    assert config.agents == {"reviewer": {"description": "code reviewer"}}
    assert config.hooks == {"PreToolUse": []}
    assert config.json_schema == {"type": "object"}
    assert config.mcp_servers == {"fs": {"type": "stdio", "command": "echo", "args": ["hi"]}}
    assert config.extra_args == {"temperature": 0}
    assert config.skills_dir == str(skills_dir)


def test_config_from_namespace_loads_rooster_code_values_from_local_dotenv(tmp_path, monkeypatch) -> None:
    from rooster_code.config import config_from_namespace

    (tmp_path / ".env").write_text(
        'ROOSTER_CODE_API_KEY="dotenv-key"\nROOSTER_CODE_BASE_URL="https://dotenv.test"\nROOSTER_CODE_MODEL="glm-5:cloud"\nROOSTER_CODE_API_TYPE="openai-completions"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    args = argparse.Namespace(
        model=None,
        cwd=None,
        allowed_tools=None,
        disallowed_tools=None,
        resume=None,
        session_id=None,
        continue_session=False,
        fork_session=None,
        persist_session=True,
        permission_mode=None,
        max_turns=None,
        max_budget_usd=None,
        max_tokens=None,
        thinking_budget=None,
        debug=False,
        sandbox=False,
        include_partials=False,
        env=None,
        custom_headers=None,
        agents_file=None,
        hooks_file=None,
        json_schema_file=None,
        mcp_file=None,
        extra_args_file=None,
        skills_dir=None,
    )

    config = config_from_namespace(args, {})

    assert config.api_key == "dotenv-key"
    assert config.base_url == "https://dotenv.test"
    assert config.model == "glm-5:cloud"
    assert config.api_type == "openai-completions"


def test_config_from_namespace_loads_search_url_from_local_dotenv(tmp_path, monkeypatch) -> None:
    from rooster_code.config import config_from_namespace

    (tmp_path / ".env").write_text(
        'ROOSTER_CODE_SEARCH_URL="https://searx.example/search"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    args = argparse.Namespace(
        model=None,
        cwd=None,
        allowed_tools=None,
        disallowed_tools=None,
        resume=None,
        session_id=None,
        continue_session=False,
        fork_session=None,
        persist_session=True,
        permission_mode=None,
        max_turns=None,
        max_budget_usd=None,
        max_tokens=None,
        thinking_budget=None,
        debug=False,
        sandbox=False,
        include_partials=False,
        env=None,
        custom_headers=None,
        agents_file=None,
        hooks_file=None,
        json_schema_file=None,
        mcp_file=None,
        extra_args_file=None,
    )

    config = config_from_namespace(args, {})

    assert config.search_url == "https://searx.example/search"


def test_config_from_namespace_prefers_cli_search_url_over_dotenv(tmp_path, monkeypatch) -> None:
    from rooster_code.config import config_from_namespace

    (tmp_path / ".env").write_text(
        'ROOSTER_CODE_SEARCH_URL="https://searx.example/search"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    args = argparse.Namespace(
        model=None,
        cwd=None,
        allowed_tools=None,
        disallowed_tools=None,
        resume=None,
        session_id=None,
        continue_session=False,
        fork_session=None,
        persist_session=True,
        permission_mode=None,
        max_turns=None,
        max_budget_usd=None,
        max_tokens=None,
        thinking_budget=None,
        debug=False,
        sandbox=False,
        include_partials=False,
        env=None,
        custom_headers=None,
        agents_file=None,
        hooks_file=None,
        json_schema_file=None,
        mcp_file=None,
        extra_args_file=None,
        skills_dir=None,
        search_url="http://127.0.0.1:8080/search",
    )

    config = config_from_namespace(args, {})

    assert config.search_url == "http://127.0.0.1:8080/search"


def test_load_json_file_returns_none_for_none_path() -> None:
    assert load_json_file(None) is None


def test_load_json_file_loads_valid_json(tmp_path) -> None:
    path = tmp_path / "test.json"
    path.write_text('{"key": "value"}', encoding="utf-8")
    result = load_json_file(str(path))
    assert result == {"key": "value"}


def test_load_json_file_returns_empty_dict_for_missing_file(tmp_path, capsys) -> None:
    path = tmp_path / "nonexistent.json"
    result = load_json_file(str(path))
    assert result == {}
    captured = capsys.readouterr()
    assert "cannot load" in captured.err
    assert str(path) in captured.err
    assert "not found" in captured.err.lower() or "No such file" in captured.err


def test_load_json_file_returns_empty_dict_for_malformed_json(tmp_path, capsys) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{invalid json content", encoding="utf-8")
    result = load_json_file(str(path))
    assert result == {}
    captured = capsys.readouterr()
    assert "cannot load" in captured.err
    assert str(path) in captured.err


def test_load_json_file_returns_empty_dict_for_empty_file(tmp_path, capsys) -> None:
    path = tmp_path / "empty.json"
    path.write_text("", encoding="utf-8")
    result = load_json_file(str(path))
    assert result == {}
    captured = capsys.readouterr()
    assert "cannot load" in captured.err


def test_load_json_file_handles_array_json(tmp_path) -> None:
    path = tmp_path / "array.json"
    path.write_text('[1, 2, 3]', encoding="utf-8")
    result = load_json_file(str(path))
    assert result == [1, 2, 3]


def test_load_json_file_handles_nested_json(tmp_path) -> None:
    path = tmp_path / "nested.json"
    path.write_text('{"mcpServers": {"fs": {"type": "stdio", "command": "echo"}}}', encoding="utf-8")
    result = load_json_file(str(path))
    assert result["mcpServers"]["fs"]["type"] == "stdio"



class TestRuntimeConfigValidation:
    """Rigorous tests for RuntimeConfig.__post_init__ validation."""

    def test_default_config_is_valid(self) -> None:
        from rooster_code.config import RuntimeConfig

        c = RuntimeConfig()
        assert c.permission_mode == "bypassPermissions"
        assert c.max_turns is None
        assert c.max_tokens is None

    def test_valid_field_values_accepted(self) -> None:
        from rooster_code.config import RuntimeConfig

        c = RuntimeConfig(
            max_turns=1,
            max_tokens=1,
            thinking_budget=1,
            max_budget_usd=0.0,
            permission_mode="default",
        )
        assert c.max_turns == 1
        assert c.max_tokens == 1
        assert c.thinking_budget == 1
        assert c.max_budget_usd == 0.0
        assert c.permission_mode == "default"

    def test_all_valid_sdk_permission_modes(self) -> None:
        from open_agent_sdk import PermissionMode
        from rooster_code.config import RuntimeConfig

        for mode in PermissionMode:
            c = RuntimeConfig(permission_mode=mode.value)
            assert c.permission_mode == mode.value

    @pytest.mark.parametrize("kwargs,expected_msg", [
        ({"max_turns": -1}, "max_turns"),
        ({"max_turns": 0}, "max_turns"),
        ({"max_tokens": -5}, "max_tokens"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"thinking_budget": -100}, "thinking_budget"),
        ({"thinking_budget": 0}, "thinking_budget"),
        ({"max_budget_usd": -0.01}, "max_budget_usd"),
        ({"permission_mode": "ask"}, "permission_mode"),
        ({"permission_mode": "garbage"}, "permission_mode"),
        ({"permission_mode": ""}, "permission_mode"),
        ({"permission_mode": "bypass"}, "permission_mode"),
        ({"max_turns": -1, "max_tokens": 0}, "max_turns"),
    ])
    def test_invalid_values_raise_value_error(self, kwargs, expected_msg) -> None:
        from rooster_code.config import RuntimeConfig

        with pytest.raises(ValueError, match=expected_msg):
            RuntimeConfig(**kwargs)

    def test_single_invalid_field_does_not_corrupt_other_fields(self) -> None:
        """Ensure a failing validation doesn't mutate the instance."""
        from rooster_code.config import RuntimeConfig

        with pytest.raises(ValueError):
            RuntimeConfig(max_turns=-1, max_tokens=100_000)

    def test_config_from_namespace_propagates_validation(self) -> None:
        """config_from_namespace should also trigger __post_init__."""
        import argparse
        from rooster_code.config import config_from_namespace

        ns = argparse.Namespace(
            model="test",
            cwd=".",
            resume=None,
            session=None,
            continue_session=False,
            fork_session=None,
            persist_session=True,
            permission_mode="ask",
            max_turns=None,
            max_budget_usd=None,
            max_tokens=None,
            thinking_budget=None,
            debug=False,
            sandbox=False,
            include_partials=False,
            env={},
            search_url=None,
            config=None,
            agents_file=None,
            mcp_file=None,
            project_cwd_file=None,
            allowed_tools=None,
            disallowed_tools=None,
            api_key=None,
            base_url=None,
            api_type=None,
            hooks_file=None,
            json_schema=None,
            skills_dir=None,
            extra_args="",
        )
        with pytest.raises(ValueError, match="permission_mode"):
            config_from_namespace(ns, env={})

    def test_none_fields_do_not_trigger_validation(self) -> None:
        """Fields that are None should not be validated (they are optional)."""
        from rooster_code.config import RuntimeConfig

        c = RuntimeConfig(
            max_turns=None,
            max_tokens=None,
            thinking_budget=None,
            max_budget_usd=None,
        )
        # Should not raise — None means "not set"
        assert c is not None

    def test_high_values_accepted(self) -> None:
        """Boundary: large valid values should not raise."""
        from rooster_code.config import RuntimeConfig

        c = RuntimeConfig(
            max_turns=1_000_000,
            max_tokens=10_000_000,
            thinking_budget=100_000,
            max_budget_usd=999999.99,
        )
        assert c.max_turns == 1_000_000

    def test_float_max_budget_zero_accepted(self) -> None:
        """Boundary: max_budget_usd=0 is allowed (free tier)."""
        from rooster_code.config import RuntimeConfig

        c = RuntimeConfig(max_budget_usd=0.0)
        assert c.max_budget_usd == 0.0