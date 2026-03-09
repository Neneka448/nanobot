"""Tool for reading lightweight stats about the current session."""

import json
from typing import Any

from nanobot.agent.tools.base import Tool
from nanobot.session.manager import SessionManager


class SessionStatsTool(Tool):
    """Expose current session token stats to the agent."""

    def __init__(self, session_manager: SessionManager, session_key: str = ""):
        self._sessions = session_manager
        self._session_key = session_key

    def set_context(self, channel: str, chat_id: str) -> None:
        """Set the current session context from channel/chat_id."""
        self._session_key = f"{channel}:{chat_id}"

    def set_session_key(self, session_key: str) -> None:
        """Set the exact current session key."""
        self._session_key = session_key

    @property
    def name(self) -> str:
        return "session_stats"

    @property
    def description(self) -> str:
        return "Get lightweight stats for the current conversation session, including total token usage."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
        }

    async def execute(self, **kwargs: Any) -> str:
        if not self._session_key:
            return "Error: no current session context"

        session = self._sessions.get_or_create(self._session_key)
        return json.dumps(
            {
                "session_key": self._session_key,
                "total_tokens": int(session.metadata.get("total_tokens", 0) or 0),
            },
            ensure_ascii=False,
        )
