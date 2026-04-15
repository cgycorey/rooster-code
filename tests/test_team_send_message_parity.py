import asyncio

from open_agent_sdk.types import ToolContext

from rooster_code.team import AgentPool, TeamManager, TeamSendMessageTool


def test_send_message_tool_preserves_sdk_message_type():
    async def _run():
        manager = TeamManager()
        manager._active = True
        manager._member_definitions = {"reviewer": {"description": "reviews"}}
        pool = AgentPool()
        pool._members["reviewer"] = object()
        pool._mailboxes["reviewer"] = asyncio.Queue()
        pool._locks["reviewer"] = asyncio.Lock()
        manager._pool = pool

        tool = TeamSendMessageTool(manager, sender_name="builder")
        result = await tool.call(
            {"to": "reviewer", "content": "please stop after this task", "type": "shutdown_request"},
            ToolContext(cwd=".", env={}),
        )

        assert not result.is_error
        assert "reviewer" in str(result.content)
        message = pool._mailboxes["reviewer"].get_nowait()
        assert message == {
            "type": "shutdown_request",
            "from": "builder",
            "content": "please stop after this task",
        }

    asyncio.run(_run())


def test_send_message_tool_supports_sdk_broadcast_semantics():
    async def _run():
        manager = TeamManager()
        manager._active = True
        manager._team_name = "dev-team"
        manager._member_definitions = {
            "builder": {"description": "builds"},
            "reviewer": {"description": "reviews"},
        }
        pool = AgentPool()
        for member_name in manager._member_definitions:
            pool._members[member_name] = object()
            pool._mailboxes[member_name] = asyncio.Queue()
            pool._locks[member_name] = asyncio.Lock()
        manager._pool = pool

        tool = TeamSendMessageTool(manager, sender_name="orchestrator")
        result = await tool.call({"to": "*", "content": "sync in ten minutes"}, ToolContext(cwd=".", env={}))

        assert not result.is_error
        assert result.content == "Message broadcast to all agents."
        assert pool._mailboxes["builder"].get_nowait() == {
            "type": "text",
            "from": "orchestrator",
            "content": "sync in ten minutes",
        }
        assert pool._mailboxes["reviewer"].get_nowait() == {
            "type": "text",
            "from": "orchestrator",
            "content": "sync in ten minutes",
        }

    asyncio.run(_run())
