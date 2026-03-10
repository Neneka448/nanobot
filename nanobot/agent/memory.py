"""Memory system for persistent agent memory."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING, Any
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

_SHORT_TERM_EXTRACT_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "short_term_events",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "events": {
                    "type": "array",
                    "description": "Distinct tasks or events extracted from the conversation.",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "scene": {
                                "type": "string",
                                "description": "One-line label for this task or problem domain.",
                            },
                            "what": {
                                "type": "string",
                                "description": "What was done or attempted.",
                            },
                            "who": {
                                "type": "string",
                                "description": "Who initiated or was involved.",
                            },
                            "why": {
                                "type": "string",
                                "description": "The goal or motivation behind the task.",
                            },
                            "how": {
                                "type": "string",
                                "description": "The approach or method used.",
                            },
                            "result": {
                                "type": "string",
                                "description": "Outcome: explicitly state success or failure and the reason.",
                            },
                        },
                        "required": ["scene", "what", "who", "why", "how", "result"],
                    },
                }
            },
            "required": ["events"],
        },
    },
}

_MIN_GROUP_REFERENCES = 4

_DEFAULT_MEMORYSTORE_MEMORY = """
# Long-term Memory

This file stores important information that should persist across sessions.

## User Information

(Important facts about the user)

## Preferences

(User preferences learned over time)

## Project Context

(Information about ongoing projects)

## Important Notes

(Things to remember)

---

*This file is automatically updated by nanobot when important information should be remembered.*
"""

_DEFAULT_LONG_TERM_MEMORY = """
# Long-term Memory

This file stores strengthened reusable knowledge consolidated from repeated experiences.

- Durable user preferences
- Stable project context
- Repeatedly validated working knowledge

---

*This file is updated by nanobot when repeated short-term experiences are promoted into long-term memory.*
"""

_EMPTY_MEMORY_FILE = """
"""

_DEFAULT_MEMORYSTORE_SKILL = """
---
name: memory
description: Two-layer memory system with grep-based recall.
always: true
---

# Memory

## Structure

- `memory/MEMORY.md` — Long-term facts (preferences, project context, relationships). Always loaded into your context.
- `memory/HISTORY.md` — Append-only event log. NOT loaded into context. Search it with grep-style tools or in-memory filters. Each entry starts with [YYYY-MM-DD HH:MM].

## Search Past Events

Choose the search method based on file size:

- Small `memory/HISTORY.md`: use `read_file`, then search in-memory
- Large or long-lived `memory/HISTORY.md`: use the `exec` tool for targeted search

Prefer targeted command-line search for large history files.

## When to Update MEMORY.md

Write important facts immediately using `edit_file` or `write_file`:
- User preferences ("I prefer dark mode")
- Project context ("The API uses OAuth2")
- Relationships ("Alice is the project lead")

## Auto-consolidation

Old conversations are automatically summarized and appended to HISTORY.md when the session grows large. Long-term facts are extracted to MEMORY.md. You don't need to manage this.
"""

_DEFAULT_LONG_SHORT_TERM_SKILL = """
---
name: memory
description: Long/short-term memory with repetition-based consolidation.
always: true
---

# Memory

Your system prompt contains a `<memories>` block with experience distilled from past sessions. The memory system is fully automated — **do not read, write, or modify any memory files**.

Every time you receive a request, silently run the following COT before forming your response.

## COT Format

```
<scene>One sentence: what task or domain is the user presenting right now?</scene>
<memory_check>Does any group in <memories> cover the same scene? Quote the key part. If none match, write: no relevant memory.</memory_check>
<decision>APPLY or IGNORE — one sentence explaining why.</decision>
<plan>Your intended approach. If APPLY, state specifically what you are borrowing from memory and why it fits.</plan>
```

## Few-Shot Examples

### Example 1 — Exact match, fully applied

User: "The import still fails after I added the new module."

```
<scene>Diagnosing a Python ImportError after adding a new package to the project.</scene>
<memory_check>Memory group "Python import resolution" notes: editable installs require re-running `pip install -e .` after adding packages; this project uses pyproject.toml with dynamic deps.</memory_check>
<decision>APPLY — the scene is identical and the guidance is concrete and project-specific.</decision>
<plan>First verify pyproject.toml lists the new package, then check whether the editable install needs refreshing before investigating other causes.</plan>
```

### Example 2 — Partial match, applied with caution

User: "Getting a TypeScript error in this new file I just created."

```
<scene>TypeScript type error in a newly created source file.</scene>
<memory_check>Memory group "TypeScript strict mode errors" notes this project has strictNullChecks=true and recommends optional chaining patterns for nullable values. The file is new, so the project-level config still applies even though the specific logic is unfamiliar.</memory_check>
<decision>APPLY partially — project tsconfig context is directly relevant; I will not over-constrain analysis of the new logic itself.</decision>
<plan>Read tsconfig to confirm strict settings, then address the type error using the optional-chaining patterns from memory where they naturally fit, without forcing them elsewhere.</plan>
```

### Example 3 — No relevant memory, ignored

User: "Write a haiku about rain."

```
<scene>Creative writing — composing a haiku.</scene>
<memory_check>No relevant memory. All groups cover coding tasks and project context; none relate to creative writing.</memory_check>
<decision>IGNORE — memory is not applicable to this task.</decision>
<plan>Treat as a novel request and compose the haiku directly.</plan>
```
"""


def _render_default_content(content: str) -> str:
    normalized = dedent(content).lstrip("\n")
    if not normalized:
        return ""
    return normalized.rstrip() + "\n"


def _write_default_file(
    workspace: Path, destination: Path, content: str, added: list[str]
) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_render_default_content(content), encoding="utf-8")
    added.append(str(destination.relative_to(workspace)))


class MemoryStore:
    """Two-layer memory: MEMORY.md (long-term facts) + HISTORY.md (grep-searchable log)."""

    def __init__(self, workspace: Path):
        self.workspace = workspace
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
        if not long_term:
            return ""
        return f"<memories>\n{long_term.strip()}\n</memories>"

    def get_memory_locations(self, workspace_path: str) -> str:
        return (
            f"- Long-term memory: {workspace_path}/memory/MEMORY.md "
            "(write reusable knowledge here)\n"
            f"- History log: {workspace_path}/memory/HISTORY.md "
            "(grep-searchable legacy log). Each entry starts with [YYYY-MM-DD HH:MM]."
        )

    def initialize_memory_files(self) -> list[str]:
        added: list[str] = []
        _write_default_file(
            self.workspace, self.memory_file, _DEFAULT_MEMORYSTORE_MEMORY, added
        )
        _write_default_file(
            self.workspace, self.history_file, _EMPTY_MEMORY_FILE, added
        )
        return added

    def initialize_memory_skill(self) -> list[str]:
        added: list[str] = []
        _write_default_file(
            self.workspace,
            self.workspace / "skills" / "memory" / "SKILL.md",
            _DEFAULT_MEMORYSTORE_SKILL,
            added,
        )
        return added

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
        self.workspace = workspace
        self.memory_dir = ensure_dir(workspace / "memory")
        self.history_file = self.memory_dir / "HISTORY.md"
        self.long_term_memory_file = self.memory_dir / "LONG_TERM_MEMORY.md"
        self.short_term_memory_file = self.memory_dir / "SHORT_TERM_MEMORY.md"
        self.short_term_window = 50

    def get_memory_context(self) -> str:
        long_term = self.read(self.long_term_memory_file)
        if not long_term:
            return ""
        return f"<memories>\n{long_term.strip()}\n</memories>"

    def get_memory_locations(self, workspace_path: str) -> str:
        return (
            f"- Long-term memory: {workspace_path}/memory/LONG_TERM_MEMORY.md "
            "(write strengthened reusable knowledge here)\n"
            f"- Short-term memory: {workspace_path}/memory/SHORT_TERM_MEMORY.md "
            "(store recent experiences before consolidation)\n"
            f"- History log: {workspace_path}/memory/HISTORY.md "
            "(grep-searchable legacy log). Each entry starts with [YYYY-MM-DD HH:MM]."
        )

    def initialize_memory_files(self) -> list[str]:
        added: list[str] = []
        _write_default_file(
            self.workspace,
            self.long_term_memory_file,
            _DEFAULT_LONG_TERM_MEMORY,
            added,
        )
        _write_default_file(
            self.workspace,
            self.short_term_memory_file,
            _EMPTY_MEMORY_FILE,
            added,
        )
        _write_default_file(
            self.workspace, self.history_file, _EMPTY_MEMORY_FILE, added
        )
        return added

    def initialize_memory_skill(self) -> list[str]:
        added: list[str] = []
        _write_default_file(
            self.workspace,
            self.workspace / "skills" / "memory" / "SKILL.md",
            _DEFAULT_LONG_SHORT_TERM_SKILL,
            added,
        )
        return added

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

    async def _extract_short_term_events(
        self,
        messages: list[dict[str, Any]],
        provider: LLMProvider,
        model: str,
    ) -> list[dict[str, Any]]:
        lines = []
        for m in messages:
            content = m.get("content")
            if not content:
                continue
            role = m.get("role", "unknown").upper()
            ts = str(m.get("timestamp", ""))[:16]
            tools = (
                f" [tools: {', '.join(m['tools_used'])}]" if m.get("tools_used") else ""
            )
            lines.append(f"[{ts}] {role}{tools}: {content}")

        if not lines:
            return []

        prompt = f"""Extract all distinct tasks and events from this conversation as structured 5W1H records.

Rules:
- One event covers one coherent task, request, or decision.
- Fill every field concisely. For unknown values, write a brief inference.
- result must explicitly state success or failure and the reason why.
- Return JSON only.

## Conversation
{chr(10).join(lines)}"""

        response = await provider.chat(
            messages=[
                {
                    "role": "system",
                    "content": "You are a memory extraction agent. Extract structured 5W1H events from the conversation and return strict JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            model=model,
            response_format=_SHORT_TERM_EXTRACT_RESPONSE_FORMAT,
        )

        if not response.content:
            logger.warning("Short-term event extraction: empty response")
            return []

        try:
            payload = json.loads(response.content)
            events = payload.get("events", [])
            if not isinstance(events, list):
                return []
            return [e for e in events if isinstance(e, dict)]
        except json.JSONDecodeError:
            logger.warning("Short-term event extraction: invalid JSON")
            return []

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
            if not archive_all:
                # Sliding window: extract unconsolidated messages before keep_count
                # into SHORT_TERM_MEMORY as 5W1H events, then advance the pointer.
                keep_count = memory_window // 2
                if len(session.messages) <= keep_count:
                    return True
                new_consolidated = len(session.messages) - keep_count
                if new_consolidated <= session.last_consolidated:
                    return True
                old_messages = session.messages[
                    session.last_consolidated : new_consolidated
                ]
                if not old_messages:
                    return True

                logger.info(
                    "Long/short-term memory: archiving {} messages to short-term, keep={}",
                    len(old_messages),
                    keep_count,
                )

                events = await self._extract_short_term_events(
                    old_messages, provider, model
                )
                if events:
                    session_id = uuid.uuid4().hex[:8]
                    short_term = self.split_entries(
                        self.read(self.short_term_memory_file)
                    )
                    for event in events:
                        event["session_id"] = session_id
                        short_term.append(json.dumps(event, ensure_ascii=False))

                    if len(short_term) > self.short_term_window:
                        short_term = await self.consolidate_short_term(
                            short_term, provider, model
                        )

                    self.write(self.short_term_memory_file, "\n\n".join(short_term))

                session.last_consolidated = new_consolidated
                return True

            # archive_all=True: triggered by /new — full session archival
            old_messages = session.messages
            if not old_messages:
                return True

            logger.info(
                "Long/short-term memory consolidation (archive_all): {} messages",
                len(old_messages),
            )

            session_id = uuid.uuid4().hex[:8]
            archived_at = datetime.now().isoformat(timespec="seconds")

            # 1. Write full raw session to HISTORY.md
            history_entry = json.dumps(
                {
                    "session_id": session_id,
                    "archived_at": archived_at,
                    "message_count": len(old_messages),
                    "messages": old_messages,
                },
                ensure_ascii=False,
            )
            self.append_content(self.history_file, history_entry)

            # 2. Extract 5W1H events → SHORT_TERM_MEMORY.md (tagged with session_id)
            events = await self._extract_short_term_events(
                old_messages, provider, model
            )
            if events:
                short_term = self.split_entries(self.read(self.short_term_memory_file))
                for event in events:
                    event["session_id"] = session_id
                    short_term.append(json.dumps(event, ensure_ascii=False))

                if len(short_term) > self.short_term_window:
                    short_term = await self.consolidate_short_term(
                        short_term, provider, model
                    )

                self.write(self.short_term_memory_file, "\n\n".join(short_term))

            session.last_consolidated = 0
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
