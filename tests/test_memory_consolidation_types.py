"""Test MemoryStore.consolidate() handles non-string tool call arguments.

Regression test for https://github.com/HKUDS/nanobot/issues/1042
When memory consolidation receives dict values instead of strings from the LLM
tool call response, it should serialize them to JSON instead of raising TypeError.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.memory import LongShortTermMemory, MemoryStore
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMResponse, ToolCallRequest


def _make_session(message_count: int = 30, memory_window: int = 50):
    """Create a mock session with messages."""
    session = MagicMock()
    session.messages = [
        {"role": "user", "content": f"msg{i}", "timestamp": "2026-01-01 00:00"}
        for i in range(message_count)
    ]
    session.last_consolidated = 0
    return session


def _make_long_short_term_session(contents: list[str]):
    """Create a mock session with concrete message contents."""
    session = MagicMock()
    session.messages = [
        {
            "role": "user",
            "content": content,
            "timestamp": f"2026-01-01 00:00:0{index}",
        }
        for index, content in enumerate(contents)
    ]
    session.last_consolidated = 0
    return session


def test_agent_loop_defaults_to_memory_store(tmp_path: Path) -> None:
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"

    loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        memory_window=10,
    )

    assert isinstance(loop.memory, MemoryStore)


def test_agent_loop_uses_explicit_long_short_term_memory(tmp_path: Path) -> None:
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    memory = LongShortTermMemory(tmp_path)

    loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        memory_window=10,
        memory=memory,
    )

    assert loop.memory is memory
    assert loop.context.memory is memory


def _make_tool_response(history_entry, memory_update):
    """Create an LLMResponse with a save_memory tool call."""
    return LLMResponse(
        content=None,
        tool_calls=[
            ToolCallRequest(
                id="call_1",
                name="save_memory",
                arguments={
                    "history_entry": history_entry,
                    "memory_update": memory_update,
                },
            )
        ],
    )


class TestMemoryConsolidationTypeHandling:
    """Test that consolidation handles various argument types correctly."""

    @pytest.mark.asyncio
    async def test_string_arguments_work(self, tmp_path: Path) -> None:
        """Normal case: LLM returns string arguments."""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()
        provider.chat = AsyncMock(
            return_value=_make_tool_response(
                history_entry="[2026-01-01] User discussed testing.",
                memory_update="# Memory\nUser likes testing.",
            )
        )
        session = _make_session(message_count=60)

        result = await store.consolidate(
            session, provider, "test-model", memory_window=50
        )

        assert result is True
        assert store.history_file.exists()
        assert "[2026-01-01] User discussed testing." in store.history_file.read_text()
        assert "User likes testing." in store.memory_file.read_text()

    @pytest.mark.asyncio
    async def test_dict_arguments_serialized_to_json(self, tmp_path: Path) -> None:
        """Issue #1042: LLM returns dict instead of string — must not raise TypeError."""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()
        provider.chat = AsyncMock(
            return_value=_make_tool_response(
                history_entry={
                    "timestamp": "2026-01-01",
                    "summary": "User discussed testing.",
                },
                memory_update={"facts": ["User likes testing"], "topics": ["testing"]},
            )
        )
        session = _make_session(message_count=60)

        result = await store.consolidate(
            session, provider, "test-model", memory_window=50
        )

        assert result is True
        assert store.history_file.exists()
        history_content = store.history_file.read_text()
        parsed = json.loads(history_content.strip())
        assert parsed["summary"] == "User discussed testing."

        memory_content = store.memory_file.read_text()
        parsed_mem = json.loads(memory_content)
        assert "User likes testing" in parsed_mem["facts"]

    @pytest.mark.asyncio
    async def test_string_arguments_as_raw_json(self, tmp_path: Path) -> None:
        """Some providers return arguments as a JSON string instead of parsed dict."""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()

        # Simulate arguments being a JSON string (not yet parsed)
        response = LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="call_1",
                    name="save_memory",
                    arguments=json.dumps(
                        {
                            "history_entry": "[2026-01-01] User discussed testing.",
                            "memory_update": "# Memory\nUser likes testing.",
                        }
                    ),
                )
            ],
        )
        provider.chat = AsyncMock(return_value=response)
        session = _make_session(message_count=60)

        result = await store.consolidate(
            session, provider, "test-model", memory_window=50
        )

        assert result is True
        assert "User discussed testing." in store.history_file.read_text()

    @pytest.mark.asyncio
    async def test_no_tool_call_returns_false(self, tmp_path: Path) -> None:
        """When LLM doesn't use the save_memory tool, return False."""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()
        provider.chat = AsyncMock(
            return_value=LLMResponse(
                content="I summarized the conversation.", tool_calls=[]
            )
        )
        session = _make_session(message_count=60)

        result = await store.consolidate(
            session, provider, "test-model", memory_window=50
        )

        assert result is False
        assert not store.history_file.exists()

    @pytest.mark.asyncio
    async def test_skips_when_few_messages(self, tmp_path: Path) -> None:
        """Consolidation should be a no-op when messages < keep_count."""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()
        session = _make_session(message_count=10)

        result = await store.consolidate(
            session, provider, "test-model", memory_window=50
        )

        assert result is True
        provider.chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_list_arguments_extracts_first_dict(self, tmp_path: Path) -> None:
        """Some providers return arguments as a list - extract first element if it's a dict."""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()

        # Simulate arguments being a list containing a dict
        response = LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="call_1",
                    name="save_memory",
                    arguments=[
                        {
                            "history_entry": "[2026-01-01] User discussed testing.",
                            "memory_update": "# Memory\nUser likes testing.",
                        }
                    ],
                )
            ],
        )
        provider.chat = AsyncMock(return_value=response)
        session = _make_session(message_count=60)

        result = await store.consolidate(
            session, provider, "test-model", memory_window=50
        )

        assert result is True
        assert "User discussed testing." in store.history_file.read_text()
        assert "User likes testing." in store.memory_file.read_text()

    @pytest.mark.asyncio
    async def test_list_arguments_empty_list_returns_false(
        self, tmp_path: Path
    ) -> None:
        """Empty list arguments should return False."""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()

        response = LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="call_1",
                    name="save_memory",
                    arguments=[],
                )
            ],
        )
        provider.chat = AsyncMock(return_value=response)
        session = _make_session(message_count=60)

        result = await store.consolidate(
            session, provider, "test-model", memory_window=50
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_list_arguments_non_dict_content_returns_false(
        self, tmp_path: Path
    ) -> None:
        """List with non-dict content should return False."""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()

        response = LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="call_1",
                    name="save_memory",
                    arguments=["string", "content"],
                )
            ],
        )
        provider.chat = AsyncMock(return_value=response)
        session = _make_session(message_count=60)

        result = await store.consolidate(
            session, provider, "test-model", memory_window=50
        )

        assert result is False


class TestLongShortTermMemoryConsolidation:
    """Test the structured long/short-term consolidation flow."""

    @pytest.mark.asyncio
    async def test_archive_all_uses_xml_prompt_and_prunes_promoted_entries(
        self, tmp_path: Path
    ) -> None:
        """Valid groups should append to long-term memory and be removed from short-term."""
        store = LongShortTermMemory(tmp_path)
        provider = AsyncMock()
        provider.chat = AsyncMock(
            return_value=LLMResponse(
                content=json.dumps(
                    {
                        "groups": [
                            {
                                "scene": "Memory prompt debugging",
                                "experience": [
                                    "What: consolidated repeated debugging work; How: compared six traces; Result: isolated the recurring pattern; Why: repeated cases justify strengthening.",
                                ],
                                "narrative": [
                                    "Who: the agent; What: investigated repeated memory issues; When: across recent sessions; Where: memory consolidation flow; Why: preserve reusable know-how; How: reviewed traces, grouped similar entries, and retained the durable pattern.",
                                ],
                                "references": [0, 1, 2, 3],
                            },
                            {
                                "scene": "Too small to keep",
                                "experience": ["What: x; How: y; Result: z; Why: q."],
                                "narrative": [
                                    "Who: x; What: y; When: z; Where: q; Why: w; How: e."
                                ],
                                "references": [4, 5, 5],
                            },
                            {
                                "scene": "Overlapping group",
                                "experience": ["What: x; How: y; Result: z; Why: q."],
                                "narrative": [
                                    "Who: x; What: y; When: z; Where: q; Why: w; How: e."
                                ],
                                "references": [2, 3, 4, 5],
                            },
                        ]
                    },
                    ensure_ascii=False,
                )
            )
        )
        session = _make_long_short_term_session(
            [
                "Fix <xml> escaping in memory prompt",
                "Compare similar prompt outputs",
                "Extract repeated debugging scene",
                "Write What-How-Result-Why summary",
                "Keep unique memory entry A",
                "Keep unique memory entry B",
            ]
        )

        result = await store.consolidate(
            session, provider, "test-model", archive_all=True
        )

        assert result is True
        provider.chat.assert_awaited_once()

        kwargs = provider.chat.await_args.kwargs
        assert kwargs["model"] == "test-model"
        assert kwargs["response_format"]["type"] == "json_schema"

        user_prompt = kwargs["messages"][1]["content"]
        assert "<memories>" in user_prompt
        assert '<entry index="0">' in user_prompt
        assert "&lt;xml&gt;" in user_prompt

        long_term_payload = json.loads(store.read(store.long_term_memory_file))
        assert len(long_term_payload["groups"]) == 1
        assert long_term_payload["groups"][0]["scene"] == "Memory prompt debugging"
        assert long_term_payload["groups"][0]["references"] == [0, 1, 2, 3]

        short_term_entries = store.split_entries(
            store.read(store.short_term_memory_file)
        )
        assert [json.loads(entry)["content"] for entry in short_term_entries] == [
            "Keep unique memory entry A",
            "Keep unique memory entry B",
        ]

    @pytest.mark.asyncio
    async def test_invalid_json_response_keeps_short_term_entries(
        self, tmp_path: Path
    ) -> None:
        """Invalid structured output must not destroy short-term memory."""
        store = LongShortTermMemory(tmp_path)
        provider = AsyncMock()
        provider.chat = AsyncMock(return_value=LLMResponse(content="not-json"))
        session = _make_long_short_term_session(
            [
                "Entry one",
                "Entry two",
                "Entry three",
                "Entry four",
            ]
        )

        result = await store.consolidate(
            session, provider, "test-model", archive_all=True
        )

        assert result is True
        short_term_entries = store.split_entries(
            store.read(store.short_term_memory_file)
        )
        assert [json.loads(entry)["content"] for entry in short_term_entries] == [
            "Entry one",
            "Entry two",
            "Entry three",
            "Entry four",
        ]
        assert not store.long_term_memory_file.exists()
