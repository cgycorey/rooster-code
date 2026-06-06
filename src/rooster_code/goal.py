from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from rooster_code.config import save_json_file

DEFAULT_GOALS_PATH = Path.home() / ".rooster-code" / "goals.json"


@dataclass(slots=True)
class Goal:
    """A user-defined goal that persists across sessions."""
    id: str
    text: str
    status: str          # "active" or "completed"
    created_at: float
    completed_at: float | None = None


def _load_goals() -> dict[str, dict]:
    try:
        fd = os.open(str(DEFAULT_GOALS_PATH), os.O_RDONLY | os.O_NOFOLLOW)
    except OSError:
        return {}
    try:
        with open(fd, "r", encoding="utf-8", closefd=False) as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}
    finally:
        os.close(fd)
    if not isinstance(data, dict):
        return {}
    return data


def _save_goals(goals: dict[str, dict]) -> None:
    save_json_file(str(DEFAULT_GOALS_PATH), goals)


def get_active_goal() -> Goal | None:
    goals = _load_goals()
    for goal_data in goals.values():
        if isinstance(goal_data, dict) and goal_data.get("status") == "active":
            return Goal(**goal_data)
    return None


def set_goal(text: str) -> Goal:
    goals = _load_goals()
    for existing in goals.values():
        if isinstance(existing, dict) and existing.get("status") == "active":
            existing["status"] = "completed"
            existing["completed_at"] = time.time()
    goal = Goal(
        id=str(uuid.uuid4())[:8],
        text=text,
        status="active",
        created_at=time.time(),
        completed_at=None,
    )
    goals[goal.id] = {
        "id": goal.id,
        "text": goal.text,
        "status": goal.status,
        "created_at": goal.created_at,
        "completed_at": goal.completed_at,
    }
    _save_goals(goals)
    return goal


def clear_goal() -> Goal | None:
    goals = _load_goals()
    for goal_data in goals.values():
        if isinstance(goal_data, dict) and goal_data.get("status") == "active":
            goal_data["status"] = "completed"
            goal_data["completed_at"] = time.time()
            _save_goals(goals)
            return Goal(**goal_data)
    return None


def list_goals() -> list[Goal]:
    goals = _load_goals()
    result = [Goal(**g) for g in goals.values() if isinstance(g, dict)]
    result.sort(key=lambda g: g.created_at, reverse=True)
    return result


def build_goal_prompt_section() -> str:
    """Build the '# Current Goal' prompt section for the active goal, or empty string."""
    active = get_active_goal()
    if not active:
        return ""
    return (
        f"\n\n# Current Goal\n"
        f"You are working toward the following goal: {active.text}\n"
        f"Use /goal check to assess progress. Do not autonomously loop; wait for the user to check."
    )


def get_goal_check_prompt() -> str | None:
    active = get_active_goal()
    if not active:
        return None
    return (
        f"Goal check. Active goal: {active.text}\n\n"
        f"Assess whether this goal is met. Reply with YES or NO and your reasoning."
    )
