import json
from pathlib import Path

from nanobot.agent.loop import AgentLoop
from nanobot.providers.base import LLMResponse, ToolCallRequest
from nanobot.session.manager import SessionManager


def test_session_save_persists_stats(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    session = manager.get_or_create("cli:direct")
    session.add_message("user", "hello")
    session.add_message("assistant", "world")
    session.metadata["total_tokens"] = 321

    manager.save(session)

    session_path = tmp_path / "sessions" / "cli_direct.jsonl"
    first_line = session_path.read_text(encoding="utf-8").splitlines()[0]
    metadata = json.loads(first_line)

    assert metadata["stats"] == {
        "total_tokens": 321,
    }


def test_list_sessions_returns_stats(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    session = manager.get_or_create("telegram:123")
    session.add_message("user", "你好")
    session.metadata["total_tokens"] = 88
    manager.save(session)

    sessions = manager.list_sessions()

    assert len(sessions) == 1
    assert sessions[0]["key"] == "telegram:123"
    assert sessions[0]["stats"]["total_tokens"] == 88


def test_estimate_response_tokens_prefers_usage_total() -> None:
    loop = AgentLoop.__new__(AgentLoop)
    messages = [{"role": "user", "content": "hello"}]
    response = LLMResponse(content="world", usage={"total_tokens": 42})

    assert loop._estimate_response_tokens(messages, response) == 42


def test_estimate_response_tokens_falls_back_to_char_estimate() -> None:
    loop = AgentLoop.__new__(AgentLoop)
    messages = [{"role": "user", "content": "hello"}]
    response = LLMResponse(
        content="world",
        tool_calls=[ToolCallRequest(id="1", name="read_file", arguments={"path": "x"})],
    )

    estimated = loop._estimate_response_tokens(messages, response)

    assert isinstance(estimated, int)
    assert estimated > 0
