#!/usr/bin/env python3
"""
Demo: Team agents passing messages to each other.

This script demonstrates the multi-agent team messaging system:
1. Creates a team with two members (builder and reviewer)
2. Has the orchestrator dispatch tasks and send messages between members
3. Shows how messages are queued and delivered to agents
"""

import asyncio
from unittest.mock import MagicMock

from rooster_code.team import AgentPool, TeamManager, TeamSendMessageTool, patch_tool_pool


class DemoAgent:
    """Simulates an agent that receives tasks and messages."""

    def __init__(self, name: str):
        self.name = name
        self._options = MagicMock()
        self._options.append_system_prompt = ""
        self._options.abort_signal = None
        self._tool_pool = []
        self._history = []
        self._closed = False
        self._last_prompt = ""
        self._call_count = 0

    async def prompt(self, text: str, overrides=None):
        self._last_prompt = text
        self._call_count += 1
        # Simulate agent processing
        print(f"\n  [{self.name} received task #{self._call_count}]")
        if "[Message from" in text:
            # Extract message content
            lines = text.split("\n")
            for line in lines:
                if "[Message from" in line:
                    print(f"    📬 {line.strip()}")
        return MagicMock(text=f"{self.name} completed task #{self._call_count}")

    async def _initialize(self):
        pass

    async def close(self):
        self._closed = True

    def clear(self):
        self._history.clear()


async def main():
    print("=" * 60)
    print("TEAM AGENTS MESSAGING DEMO")
    print("=" * 60)

    # Setup
    manager = TeamManager()
    pool = AgentPool()

    # Create team members
    builder = DemoAgent("builder")
    reviewer = DemoAgent("reviewer")
    tester = DemoAgent("tester")

    members = ["builder", "reviewer", "tester"]

    # Register members with the pool
    for member_name, agent in [("builder", builder), ("reviewer", reviewer), ("tester", tester)]:
        pool._members[member_name] = agent
        pool._mailboxes[member_name] = asyncio.Queue()
        pool._locks[member_name] = asyncio.Lock()

    # Configure team manager
    manager._pool = pool
    manager._team_name = "demo-team"
    manager._active = True
    manager._member_definitions = {
        "builder": {"description": "Builds things"},
        "reviewer": {"description": "Reviews code"},
        "tester": {"description": "Runs tests"},
    }

    # Inject SendMessage tool into each member (as create_team would do)
    member_send_tool = TeamSendMessageTool(manager)
    for member_name in members:
        member_agent = pool._members[member_name]
        patch_tool_pool(
            member_agent,
            add_tools=[member_send_tool],
            remove_names=["TeamCreate", "TeamDelete", "Agent"]
        )
        others = ", ".join(m for m in members if m != member_name)
        member_prompt = f"You are '{member_name}', a member of team 'demo-team'. Other members: {others}. Use SendMessage to communicate."
        member_agent._options.append_system_prompt = member_prompt

    print("\n✓ Team created with members:", ", ".join(members))
    print("✓ Each member has SendMessage tool injected")
    print()

    # === Demo 1: Orchestrator sends message to a member ===
    print("-" * 60)
    print("DEMO 1: Orchestrator sends message to 'reviewer'")
    print("-" * 60)
    await manager.send_message("reviewer", "Please review the latest code changes", sender="orchestrator")
    print("✓ Message queued for reviewer")

    result = await manager.dispatch("reviewer", "Summarize your review findings")
    print(f"✓ Reviewer response: {result}")
    print()

    # === Demo 2: Member-to-member messaging ===
    print("-" * 60)
    print("DEMO 2: 'builder' sends message to 'tester'")
    print("-" * 60)

    # Simulate builder using SendMessage tool (as an AI agent would)
    from open_agent_sdk.types import ToolContext
    tool = TeamSendMessageTool(manager)

    result = await tool.call(
        {"to": "tester", "content": "Build is ready! Please run the test suite."},
        ToolContext(cwd=".", env={})
    )
    print(f"✓ SendMessage tool result: {result.content}")

    # Now dispatch to tester - they receive the message before their task
    result = await manager.dispatch("tester", "Report test results")
    print(f"✓ Tester response: {result}")
    print()

    # === Demo 3: Multiple messages queue up ===
    print("-" * 60)
    print("DEMO 3: Multiple messages queue for 'builder'")
    print("-" * 60)

    await manager.send_message("builder", "First message: review PR #123", sender="reviewer")
    await manager.send_message("builder", "Second message: update docs too", sender="tester")
    print("✓ Two messages queued for builder")

    result = await manager.dispatch("builder", "What's your status?")
    print(f"✓ Builder response: {result}")
    print()

    # === Demo 4: Mailbox drained after dispatch ===
    print("-" * 60)
    print("DEMO 4: Mailbox is drained after dispatch")
    print("-" * 60)

    result = await manager.dispatch("builder", "Any updates?")
    print(f"✓ Second dispatch (no pending messages): {result}")
    print()

    # === Summary ===
    print("=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("""
Key concepts demonstrated:
1. TeamSendMessageTool lets agents send messages to teammates
2. Messages are queued in each member's mailbox
3. When a member is dispatched, queued messages are injected into their prompt
4. Mailboxes are drained after each dispatch
5. Any agent (orchestrator or member) can send messages via SendMessage
""")

    # Cleanup
    await manager.close_team(MagicMock())
    print("✓ Team closed cleanly")


if __name__ == "__main__":
    asyncio.run(main())
