"""End-to-end verification: inter-member messaging with TeamSendMessageTool injection."""

import asyncio
from unittest.mock import MagicMock

from open_agent_sdk.types import ToolContext

from rooster_code.team import AgentPool, TeamManager, TeamSendMessageTool, patch_tool_pool


class FakeMemberAgent:
    def __init__(self, name):
        self.name = name
        self._options = MagicMock()
        self._options.append_system_prompt = ""
        self._tool_pool = []
        self._history = []
        self._closed = False
        self._last_prompt = ""

    async def prompt(self, text, overrides=None):
        self._last_prompt = text
        return MagicMock(text=f"{self.name} responded")

    async def _initialize(self):
        pass

    async def close(self):
        self._closed = True

    def clear(self):
        self._history.clear()


def test_member_tools_injected():
    manager = TeamManager()
    pool = AgentPool()

    builder = FakeMemberAgent("builder")
    reviewer = FakeMemberAgent("reviewer")

    pool._members["builder"] = builder
    pool._mailboxes["builder"] = asyncio.Queue()
    pool._locks["builder"] = asyncio.Lock()
    pool._members["reviewer"] = reviewer
    pool._mailboxes["reviewer"] = asyncio.Queue()
    pool._locks["reviewer"] = asyncio.Lock()

    manager._pool = pool
    manager._team_name = "dev-team"
    manager._active = True

    members = ["builder", "reviewer"]
    for member_name in members:
        member_agent = pool._members[member_name]
        member_send_tool = TeamSendMessageTool(manager, sender_name=member_name)
        patch_tool_pool(member_agent, add_tools=[member_send_tool], remove_names=["TeamCreate", "TeamDelete", "Agent"])
        others = ", ".join(m for m in members if m != member_name)
        member_prompt = f"You are '{member_name}', a member of team 'dev-team'. Other members: {others}. Use SendMessage to communicate with other team members."
        existing = getattr(member_agent._options, "append_system_prompt", "") or ""
        if existing:
            member_agent._options.append_system_prompt = existing + "\n\n" + member_prompt
        else:
            member_agent._options.append_system_prompt = member_prompt

    builder_tools = [t.name for t in builder._tool_pool]
    reviewer_tools = [t.name for t in reviewer._tool_pool]
    assert "SendMessage" in builder_tools, f"Builder missing SendMessage! Got: {builder_tools}"
    assert "SendMessage" in reviewer_tools, f"Reviewer missing SendMessage! Got: {reviewer_tools}"
    print("PASS: Both members have SendMessage tool")


def test_member_to_member_messaging():
    async def _run():
        manager = TeamManager()
        pool = AgentPool()

        builder = FakeMemberAgent("builder")
        reviewer = FakeMemberAgent("reviewer")

        pool._members["builder"] = builder
        pool._mailboxes["builder"] = asyncio.Queue()
        pool._locks["builder"] = asyncio.Lock()
        pool._members["reviewer"] = reviewer
        pool._mailboxes["reviewer"] = asyncio.Queue()
        pool._locks["reviewer"] = asyncio.Lock()

        manager._pool = pool
        manager._team_name = "dev-team"
        manager._active = True

        member_tools: dict[str, TeamSendMessageTool] = {}
        members = ["builder", "reviewer"]
        for member_name in members:
            member_agent = pool._members[member_name]
            member_send_tool = TeamSendMessageTool(manager, sender_name=member_name)
            member_tools[member_name] = member_send_tool
            patch_tool_pool(member_agent, add_tools=[member_send_tool], remove_names=["TeamCreate", "TeamDelete", "Agent"])
            others = ", ".join(m for m in members if m != member_name)
            member_prompt = f"You are '{member_name}', a member of team 'dev-team'. Other members: {others}. Use SendMessage to communicate with other team members."
            existing = getattr(member_agent._options, "append_system_prompt", "") or ""
            if existing:
                member_agent._options.append_system_prompt = existing + "\n\n" + member_prompt
            else:
                member_agent._options.append_system_prompt = member_prompt

        await member_tools["builder"].call({"to": "reviewer", "content": "please review my code"}, ToolContext(cwd=".", env={}))
        assert not pool._mailboxes["reviewer"].empty(), "Reviewer mailbox should have a message"

        # Dispatch to reviewer - prompt should include the queued message
        await manager.dispatch("reviewer", "review the build")
        assert "[Message from builder]: please review my code" in reviewer._last_prompt, \
            f"Expected message from builder in prompt, got: {reviewer._last_prompt}"
        assert "review the build" in reviewer._last_prompt, \
            f"Expected task in prompt, got: {reviewer._last_prompt}"

        await member_tools["reviewer"].call({"to": "builder", "content": "looks good, ship it"}, ToolContext(cwd=".", env={}))
        await manager.dispatch("builder", "deploy")
        assert "[Message from reviewer]: looks good, ship it" in builder._last_prompt, \
            f"Expected message from reviewer in prompt, got: {builder._last_prompt}"
        assert "deploy" in builder._last_prompt, \
            f"Expected task in prompt, got: {builder._last_prompt}"

        await manager.dispatch("reviewer", "another task")
        assert "[Message from builder]" not in reviewer._last_prompt, \
            f"Old message should be drained, got: {reviewer._last_prompt}"

        print("PASS: Member-to-member messaging works end-to-end")

    asyncio.run(_run())


def test_send_message_tool_from_member():
    async def _run():
        manager = TeamManager()
        manager._active = True
        pool = AgentPool()

        builder = FakeMemberAgent("builder")
        reviewer = FakeMemberAgent("reviewer")

        pool._members["builder"] = builder
        pool._mailboxes["builder"] = asyncio.Queue()
        pool._locks["builder"] = asyncio.Lock()
        pool._members["reviewer"] = reviewer
        pool._mailboxes["reviewer"] = asyncio.Queue()
        pool._locks["reviewer"] = asyncio.Lock()
        manager._pool = pool

        tool = TeamSendMessageTool(manager, sender_name="reviewer")

        result = await tool.call({"to": "builder", "content": "hey builder"}, ToolContext(cwd=".", env={}))
        assert not result.is_error
        assert "builder" in result.content

        # Verify message is in builder's mailbox
        assert not pool._mailboxes["builder"].empty()
        msg = pool._mailboxes["builder"].get_nowait()
        assert msg["from"] == "reviewer"
        assert msg["content"] == "hey builder"

        print("PASS: TeamSendMessageTool delivers messages to member mailboxes")

    asyncio.run(_run())


if __name__ == "__main__":
    test_member_tools_injected()
    test_member_to_member_messaging()
    test_send_message_tool_from_member()
    print("\nALL INTER-MEMBER MESSAGING TESTS PASSED")
