from typing import Protocol

from nanobot.providers.base import LLMProvider
from nanobot.session.manager import Session


class MemoryProtocol(Protocol):
    def get_memory_context(self) -> str:
        pass

    def get_memory_locations(self, workspace_path: str) -> str:
        pass

    def initialize_memory_files(self) -> list[str]:
        pass

    def initialize_memory_skill(self) -> list[str]:
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
