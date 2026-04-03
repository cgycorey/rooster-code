import argparse
import json

from cock_code.config import resolve_runtime_env


def test_cock_code_env_names_are_mapped() -> None:
    env = {
        "COCK_CODE_API_KEY": "key-123",
        "COCK_CODE_BASE_URL": "https://example.test",
        "COCK_CODE_MODEL": "test-model",
        "COCK_CODE_API_TYPE": "openai-completions",
    }

    resolved = resolve_runtime_env(env)

    assert resolved.api_key == "key-123"
    assert resolved.base_url == "https://example.test"
    assert resolved.model == "test-model"
    assert resolved.api_type == "openai-completions"


def test_config_from_namespace_loads_runtime_files_and_kv_pairs(tmp_path) -> None:
    from cock_code.config import config_from_namespace

    agents_file = tmp_path / "agents.json"
    hooks_file = tmp_path / "hooks.json"
    schema_file = tmp_path / "schema.json"
    mcp_file = tmp_path / "mcp.json"
    extra_args_file = tmp_path / "extra.json"

    agents_file.write_text(json.dumps({"reviewer": {"description": "code reviewer"}}), encoding="utf-8")
    hooks_file.write_text(json.dumps({"PreToolUse": []}), encoding="utf-8")
    schema_file.write_text(json.dumps({"type": "object"}), encoding="utf-8")
    mcp_file.write_text(json.dumps({"fs": {"type": "stdio", "command": "echo", "args": ["hi"]}}), encoding="utf-8")
    extra_args_file.write_text(json.dumps({"temperature": 0}), encoding="utf-8")

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
    )

    config = config_from_namespace(args, {"COCK_CODE_API_KEY": "env-key", "COCK_CODE_BASE_URL": "https://env.test"})

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


def test_config_from_namespace_loads_cock_code_values_from_local_dotenv(tmp_path, monkeypatch) -> None:
    from cock_code.config import config_from_namespace

    (tmp_path / ".env").write_text(
        'COCK_CODE_API_KEY="dotenv-key"\nCOCK_CODE_BASE_URL="https://dotenv.test"\nCOCK_CODE_MODEL="glm-5:cloud"\nCOCK_CODE_API_TYPE="openai-completions"\n',
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

    assert config.api_key == "dotenv-key"
    assert config.base_url == "https://dotenv.test"
    assert config.model == "glm-5:cloud"
    assert config.api_type == "openai-completions"
