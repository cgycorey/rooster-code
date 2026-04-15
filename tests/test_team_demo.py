"""Demo: Plan and Review agents communicating via messages."""

import asyncio

from rooster_code.team import AgentPool, TeamManager


class FakeQueryResult:
    def __init__(self, text: str = "done"):
        self.text = text


class FakeAgent:
    """Simulated LLM agent that returns a canned response."""

    def __init__(self, name: str, responses: list[str]):
        self.name = name
        self._responses = responses
        self._call_count = 0
        self._tool_pool = []

    @property
    def _options(self):
        class Opt:
            abort_signal = None
            append_system_prompt = ""
        return Opt()

    async def prompt(self, text: str):
        response = self._responses[self._call_count]
        self._call_count += 1
        print(f"\n--- {self.name} received ---")
        print(text[:500] + ("..." if len(text) > 500 else ""))
        print(f"--- {self.name} responds ---\n{response}\n")
        return FakeQueryResult(text=response)

    async def _initialize(self):
        pass

    async def close(self):
        pass

    def clear(self):
        pass


async def demo():
    print("=" * 60)
    print("DEMO: Plan Agent and Review Agent Messaging")
    print("=" * 60)

    # Set up agents config
    # Create team manager
    manager = TeamManager()

    # Create mock orchestrator
    class MockOrchestrator:
        _options = type("Options", (), {"append_system_prompt": ""})()
        _tool_pool = []

    # Create real agent pool manually for demo (bypass SDK init)
    pool = AgentPool()
    planner = FakeAgent("planner", [
        "Here's my plan:\n1. Design schema\n2. Implement API\n3. Write tests\n\nLet me send this to reviewer for feedback."
    ])
    reviewer = FakeAgent("reviewer", [
        "Plan looks good! Minor suggestion: add step 0 for requirements gathering."
    ])

    pool._members["planner"] = planner
    pool._mailboxes["planner"] = asyncio.Queue()
    pool._locks["planner"] = asyncio.Lock()
    pool._members["reviewer"] = reviewer
    pool._mailboxes["reviewer"] = asyncio.Queue()
    pool._locks["reviewer"] = asyncio.Lock()
    manager._pool = pool
    manager._active = True

    # --- Demo: Planner sends message to Reviewer ---
    print("\n>>> Step 1: Planner sends message to Reviewer")
    pool.send_message("reviewer", {
        "from": "planner",
        "content": "I've drafted a plan for the new feature. Please review and provide feedback."
    })

    print("\n>>> Step 2: Orchestrator dispatches task to Planner")
    result = await pool.dispatch("planner", "Create a project plan for a web app")
    print(f"Planner result: {result}")

    print("\n>>> Step 3: Orchestrator dispatches task to Reviewer (messages queued)")
    result = await pool.dispatch("reviewer", "Review the plan you received.")
    print(f"Reviewer result: {result}")

    # --- Demo: Reviewer responds back to Planner ---
    print("\n>>> Step 4: Reviewer sends feedback back to Planner")
    pool.send_message("planner", {
        "from": "reviewer",
        "content": "Your plan is solid! Consider adding error handling steps."
    })

    print("\n>>> Step 5: Orchestrator asks Planner to revise with feedback")
    result = await pool.dispatch("planner", "Revise your plan based on feedback.")
    print(f"Planner result: {result}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE: Messages were successfully passed between agents!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(demo())
