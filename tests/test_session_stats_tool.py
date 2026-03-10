import json
from unittest.mock import MagicMock, patch

import pytest

from nanobot.agent.tools.session_stats import SessionStatsTool
from nanobot.bus.queue import MessageBus
from nanobot.session.manager import SessionManager


@pytest.mark.asyncio
async def test_session_stats_tool_returns_current_total_tokens(tmp_path) -> None:
    manager = SessionManager(tmp_path)
    session = manager.get_or_create("cli:direct")
    session.metadata["total_tokens"] = 123

    tool = SessionStatsTool(manager)
    tool.set_session_key("cli:direct")
    result = await tool.execute()

    assert json.loads(result) == {
        "session_key": "cli:direct",
        "total_tokens": 123,
    }


@pytest.mark.asyncio
async def test_session_stats_tool_requires_context(tmp_path) -> None:
    tool = SessionStatsTool(SessionManager(tmp_path))

    result = await tool.execute()

    assert result == "Error: no current session context"


def test_agent_loop_registers_session_stats_tool(tmp_path) -> None:
    from nanobot.agent.loop import AgentLoop

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"

    with (
        patch("nanobot.agent.loop.ContextBuilder"),
        patch("nanobot.agent.loop.SubagentManager"),
    ):
        loop = AgentLoop(
            bus=bus, provider=provider, workspace=tmp_path, model="test-model"
        )

    assert loop.tools.has("session_stats")
