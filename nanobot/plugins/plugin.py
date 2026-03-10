from typing import Protocol

from nanobot.providers.base import LLMProvider
from nanobot.session.manager import Session


class MemoryProtocol(Protocol):
    def get_memory_context(self) -> str:
        pass

    async def consolidate(
        self,
        session: Session,
        provider: LLMProvider,
        model: str,
        *,
        archive_all: bool = False,
        memory_window: int = 50,
    ) -> bool:
        pass
