import time
import tempfile
from pathlib import Path
from unittest.mock import patch

import rooster_code.goal as goal


def test_set_and_get_active_goal():
    with tempfile.TemporaryDirectory() as d:
        goals_path = Path(d) / "goals.json"
        with patch.object(goal, "DEFAULT_GOALS_PATH", goals_path):
            g = goal.set_goal("Implement goal feature")
            assert g.text == "Implement goal feature"
            assert g.status == "active"
            assert g.id

            active = goal.get_active_goal()
            assert active is not None
            assert active.id == g.id


def test_set_goal_deactivates_existing():
    with tempfile.TemporaryDirectory() as d:
        goals_path = Path(d) / "goals.json"
        with patch.object(goal, "DEFAULT_GOALS_PATH", goals_path):
            g1 = goal.set_goal("First goal")
            g2 = goal.set_goal("Second goal")

            goals_list = goal.list_goals()
            statuses = {g.id: g.status for g in goals_list}
            assert statuses[g1.id] == "completed"
            assert statuses[g2.id] == "active"


def test_clear_goal():
    with tempfile.TemporaryDirectory() as d:
        goals_path = Path(d) / "goals.json"
        with patch.object(goal, "DEFAULT_GOALS_PATH", goals_path):
            goal.set_goal("Test goal")
            cleared = goal.clear_goal()
            assert cleared is not None
            assert cleared.status == "completed"
            active = goal.get_active_goal()
            assert active is None


def test_clear_without_active_goal():
    with tempfile.TemporaryDirectory() as d:
        goals_path = Path(d) / "goals.json"
        with patch.object(goal, "DEFAULT_GOALS_PATH", goals_path):
            result = goal.clear_goal()
            assert result is None


def test_list_goals_sorted():
    with tempfile.TemporaryDirectory() as d:
        goals_path = Path(d) / "goals.json"
        with patch.object(goal, "DEFAULT_GOALS_PATH", goals_path):
            goal.set_goal("Oldest")
            time.sleep(0.01)
            goal.set_goal("Middle")
            time.sleep(0.01)
            goal.set_goal("Newest")

            goals = goal.list_goals()
            assert len(goals) == 3
            assert goals[0].text == "Newest"
            assert goals[2].text == "Oldest"


def test_get_goal_check_prompt():
    with tempfile.TemporaryDirectory() as d:
        goals_path = Path(d) / "goals.json"
        with patch.object(goal, "DEFAULT_GOALS_PATH", goals_path):
            goal.set_goal("Write tests")
            prompt = goal.get_goal_check_prompt()
            assert prompt is not None
            assert "Write tests" in prompt
            assert "YES or NO" in prompt


def test_get_goal_check_prompt_no_active():
    with tempfile.TemporaryDirectory() as d:
        goals_path = Path(d) / "goals.json"
        with patch.object(goal, "DEFAULT_GOALS_PATH", goals_path):
            prompt = goal.get_goal_check_prompt()
            assert prompt is None
