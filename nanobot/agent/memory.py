"""Memory system for persistent agent memory."""

from __future__ import annotations

import json
from typing import Any
from pathlib import Path
from typing import TYPE_CHECKING
from xml.sax.saxutils import escape

from loguru import logger

from nanobot.utils.helpers import ensure_dir

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import Session


_SAVE_MEMORY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save the memory consolidation result to persistent storage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "history_entry": {
                        "type": "string",
                        "description": "A paragraph (2-5 sentences) summarizing key events/decisions/topics. "
                        "Start with [YYYY-MM-DD HH:MM]. Include detail useful for grep search.",
                    },
                    "memory_update": {
                        "type": "string",
                        "description": "Full updated long-term memory as markdown. Include all existing "
                        "facts plus new ones. Return unchanged if nothing new.",
                    },
                },
                "required": ["history_entry", "memory_update"],
            },
        },
    }
]

_SHORT_TERM_GROUPS_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "long_term_memory_groups",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "groups": {
                    "type": "array",
                    "description": "Memory groups ordered by similarity from high to low.",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "scene": {
                                "type": "string",
                                "description": "The scene or problem domain shared by the grouped memories.",
                            },
                            "experience": {
                                "type": "array",
                                "description": "Experience summaries. Every item must follow What-How-Result-Why.",
                                "items": {"type": "string"},
                            },
                            "narrative": {
                                "type": "array",
                                "description": "Experience narratives. Every item must use 5W1H and include failure reasons when the outcome was unsuccessful.",
                                "items": {"type": "string"},
                            },
                            "references": {
                                "type": "array",
                                "description": "Indices of the related short-term memory entries.",
                                "items": {"type": "integer"},
                            },
                        },
                        "required": ["scene", "experience", "narrative", "references"],
                    },
                }
            },
            "required": ["groups"],
        },
    },
}

_MIN_GROUP_REFERENCES = 4


class MemoryStore:
    """Two-layer memory: MEMORY.md (long-term facts) + HISTORY.md (grep-searchable log)."""

    def __init__(self, workspace: Path):
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"

    def read_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    def append_history(self, entry: str) -> None:
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")

    def get_memory_context(self) -> str:
        long_term = self.read_long_term()
        return f"## Long-term Memory\n{long_term}" if long_term else ""

    async def consolidate(
        self,
        session: Session,
        provider: LLMProvider,
        model: str,
        *,
        archive_all: bool = False,
        memory_window: int = 50,
    ) -> bool:
        """Consolidate old messages into MEMORY.md + HISTORY.md via LLM tool call.

        Returns True on success (including no-op), False on failure.
        """
        if archive_all:
            old_messages = session.messages
            keep_count = 0
            logger.info(
                "Memory consolidation (archive_all): {} messages", len(session.messages)
            )
        else:
            keep_count = memory_window // 2
            if len(session.messages) <= keep_count:
                return True
            if len(session.messages) - session.last_consolidated <= 0:
                return True
            old_messages = session.messages[session.last_consolidated : -keep_count]
            if not old_messages:
                return True
            logger.info(
                "Memory consolidation: {} to consolidate, {} keep",
                len(old_messages),
                keep_count,
            )

        lines = []
        for m in old_messages:
            if not m.get("content"):
                continue
            tools = (
                f" [tools: {', '.join(m['tools_used'])}]" if m.get("tools_used") else ""
            )
            lines.append(
                f"[{m.get('timestamp', '?')[:16]}] {m['role'].upper()}{tools}: {m['content']}"
            )

        current_memory = self.read_long_term()
        prompt = f"""Process this conversation and call the save_memory tool with your consolidation.

## Current Long-term Memory
{current_memory or "(empty)"}

## Conversation to Process
{chr(10).join(lines)}"""

        try:
            response = await provider.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a memory consolidation agent. Call the save_memory tool with your consolidation of the conversation.",
                    },
                    {"role": "user", "content": prompt},
                ],
                tools=_SAVE_MEMORY_TOOL,
                model=model,
            )

            if not response.has_tool_calls:
                logger.warning(
                    "Memory consolidation: LLM did not call save_memory, skipping"
                )
                return False

            args = response.tool_calls[0].arguments
            # Some providers return arguments as a JSON string instead of dict
            if isinstance(args, str):
                args = json.loads(args)
            # Some providers return arguments as a list (handle edge case)
            if isinstance(args, list):
                if args and isinstance(args[0], dict):
                    args = args[0]
                else:
                    logger.warning(
                        "Memory consolidation: unexpected arguments as empty or non-dict list"
                    )
                    return False
            if not isinstance(args, dict):
                logger.warning(
                    "Memory consolidation: unexpected arguments type {}",
                    type(args).__name__,
                )
                return False

            if entry := args.get("history_entry"):
                if not isinstance(entry, str):
                    entry = json.dumps(entry, ensure_ascii=False)
                self.append_history(entry)
            if update := args.get("memory_update"):
                if not isinstance(update, str):
                    update = json.dumps(update, ensure_ascii=False)
                if update != current_memory:
                    self.write_long_term(update)

            session.last_consolidated = (
                0 if archive_all else len(session.messages) - keep_count
            )
            logger.info(
                "Memory consolidation done: {} messages, last_consolidated={}",
                len(session.messages),
                session.last_consolidated,
            )
            return True
        except Exception:
            logger.exception("Memory consolidation failed")
            return False


class LongShortTermMemory:
    """Long/short-term memory store with repetition-based strengthening."""

    def __init__(self, workspace: Path):
        self.memory_dir = ensure_dir(workspace / "memory")
        self.history_file = self.memory_dir / "HISTORY.md"
        self.long_term_memory_file = self.memory_dir / "LONG_TERM_MEMORY.md"
        self.short_term_memory_file = self.memory_dir / "SHORT_TERM_MEMORY.md"
        self.short_term_window = 50

    def get_memory_context(self) -> str:
        long_term = self.read(self.long_term_memory_file)
        return f"## Long-term Memory\n{long_term}" if long_term else ""

    def read(self, file: Path) -> str:
        if file.exists():
            return file.read_text(encoding="utf-8")
        return ""

    def write(self, file: Path, content: str) -> None:
        file.write_text(content, encoding="utf-8")

    def append_content(self, file: Path, content: str) -> None:
        with open(file, "a", encoding="utf-8") as f:
            f.write(content.rstrip() + "\n\n")

    def split_entries(self, content: str) -> list[str]:
        return [part for part in content.split("\n\n") if part.strip()]

    def _serialize_short_term_entry(self, message: dict[str, Any]) -> str | None:
        content = message.get("content")
        if not content:
            return None

        entry: dict[str, Any] = {
            "timestamp": str(message.get("timestamp", "?"))[:19],
            "role": message.get("role", "unknown"),
            "content": content,
        }
        if message.get("tools_used"):
            entry["tools_used"] = message["tools_used"]
        if message.get("name"):
            entry["name"] = message["name"]

        return json.dumps(entry, ensure_ascii=False)

    def _render_short_term_xml(self, short_term: list[str]) -> str:
        if not short_term:
            return "<memories></memories>"

        lines = ["<memories>"]
        for index, entry in enumerate(short_term):
            lines.extend(
                [
                    f'<entry index="{index}">',
                    escape(entry),
                    "</entry>",
                ]
            )
        lines.append("</memories>")
        return "\n".join(lines)

    @staticmethod
    def _normalize_string_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [
            item.strip() for item in value if isinstance(item, str) and item.strip()
        ]

    def _normalize_groups(self, payload: Any, entry_count: int) -> list[dict[str, Any]]:
        if not isinstance(payload, dict):
            return []

        raw_groups = payload.get("groups")
        if not isinstance(raw_groups, list):
            return []

        selected_groups: list[dict[str, Any]] = []
        used_references: set[int] = set()

        for raw_group in raw_groups:
            if not isinstance(raw_group, dict):
                continue

            scene = raw_group.get("scene")
            if not isinstance(scene, str) or not scene.strip():
                continue

            experience = self._normalize_string_list(raw_group.get("experience"))
            narrative = self._normalize_string_list(raw_group.get("narrative"))
            if not experience or not narrative:
                continue

            raw_references = raw_group.get("references")
            if not isinstance(raw_references, list):
                continue

            references: list[int] = []
            seen: set[int] = set()
            for reference in raw_references:
                if not isinstance(reference, int):
                    continue
                if reference < 0 or reference >= entry_count or reference in seen:
                    continue
                seen.add(reference)
                references.append(reference)

            if len(references) < _MIN_GROUP_REFERENCES:
                continue
            if any(reference in used_references for reference in references):
                continue

            used_references.update(references)
            selected_groups.append(
                {
                    "scene": scene.strip(),
                    "experience": experience,
                    "narrative": narrative,
                    "references": references,
                }
            )

        return selected_groups

    async def consolidate(
        self,
        session: Session,
        provider: LLMProvider,
        model: str,
        *,
        archive_all: bool = False,
        memory_window: int = 50,
    ) -> bool:
        try:
            if archive_all:
                old_messages = session.messages
                keep_count = 0
                logger.info(
                    "Long/short-term memory consolidation (archive_all): {} messages",
                    len(session.messages),
                )
            else:
                keep_count = memory_window // 2
                if len(session.messages) <= keep_count:
                    return True
                if len(session.messages) - session.last_consolidated <= 0:
                    return True
                old_messages = session.messages[session.last_consolidated : -keep_count]
                if not old_messages:
                    return True
                logger.info(
                    "Long/short-term memory consolidation: {} to short-term, {} keep",
                    len(old_messages),
                    keep_count,
                )

            short_term = self.split_entries(self.read(self.short_term_memory_file))
            new_entries = [
                entry
                for entry in (
                    self._serialize_short_term_entry(message)
                    for message in old_messages
                )
                if entry
            ]
            short_term.extend(new_entries)

            if short_term and (archive_all or len(short_term) > self.short_term_window):
                short_term = await self.consolidate_short_term(
                    short_term, provider, model
                )

            self.write(self.short_term_memory_file, "\n\n".join(short_term))
            session.last_consolidated = (
                0 if archive_all else len(session.messages) - keep_count
            )
            return True
        except Exception:
            logger.exception("Long/short-term memory consolidation failed")
            return False

    async def consolidate_short_term(
        self, short_term: list[str], provider: LLMProvider, model: str
    ) -> list[str]:
        current_long_term = self.read(self.long_term_memory_file)
        prompt = f"""Analyze every short-term memory entry, compare their similarity, and group together entries that are clearly about the same repeated task, scene, or problem domain.

Rules:
- Read the short-term memories from the XML block.
- Only output groups whose references count is greater than 3.
- references must contain the entry indices from the XML.
- Keep groups ordered from highest similarity to lowest similarity.
- Each experience item must follow What-How-Result-Why.
- Each narrative item must use 5W1H and explicitly mention the failure reason when the result was unsuccessful.
- Return JSON only and follow the schema exactly.

## Current Long-term Memory
{current_long_term or "(empty)"}

## Short-term Memory
{self._render_short_term_xml(short_term)}
"""

        response = await provider.chat(
            messages=[
                {
                    "role": "system",
                    "content": "You are a memory consolidation agent that extracts repeated work scenes from short-term memory and returns strict JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            model=model,
            response_format=_SHORT_TERM_GROUPS_RESPONSE_FORMAT,
        )

        if not response.content:
            logger.warning(
                "Long/short-term memory consolidation: empty structured response"
            )
            return short_term

        try:
            payload = json.loads(response.content)
        except json.JSONDecodeError:
            logger.warning(
                "Long/short-term memory consolidation: invalid JSON response"
            )
            return short_term

        groups = self._normalize_groups(payload, len(short_term))
        if not groups:
            return short_term

        self.append_content(
            self.long_term_memory_file,
            json.dumps({"groups": groups}, ensure_ascii=False, indent=2),
        )

        promoted_references = {
            reference for group in groups for reference in group["references"]
        }
        return [
            entry
            for index, entry in enumerate(short_term)
            if index not in promoted_references
        ]
