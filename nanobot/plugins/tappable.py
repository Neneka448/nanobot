from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from nanobot.plugins.schema.schema import TapConsolidateMemoryContext

Ctx = TypeVar("Ctx")


class Tappable(ABC, Generic[Ctx]):
    def __init__(self, id: str, desc: str, config: dict[str, Any]):
        self.id: str = id
        self.description: str = desc
        self.config: dict[str, Any] = config

    @abstractmethod
    def tap(self, ctx: Ctx) -> Ctx:
        pass

    @abstractmethod
    def is_once(self) -> bool:
        pass


class TappableAsync(ABC, Generic[Ctx]):
    def __init__(self, id: str, desc: str, config: dict[str, Any]):
        self.id: str = id
        self.description: str = desc
        self.config: dict[str, Any] = config

    @abstractmethod
    async def tap(self, ctx: Ctx) -> Ctx:
        pass

    @abstractmethod
    def is_once(self) -> bool:
        pass


class Tap(Generic[Ctx]):
    def __init__(self, id: str, desc: str, config: dict[str, Any]):
        self.id: str = id
        self.description: str = desc
        self.config: dict[str, Any] = config

        self.taps: list[Tappable[Ctx]] = []

    def register(self, tappable: Tappable[Ctx]) -> None:
        self.taps.append(tappable)

    def unregister(self, tappable: Tappable[Ctx]) -> None:
        self.taps.remove(tappable)

    def tap(self, ctx: Ctx) -> Ctx:
        for tappable in self.taps:
            ctx = tappable.tap(ctx)

        return ctx


class TapAsync(Generic[Ctx]):
    def __init__(self, id: str, desc: str, config: dict[str, Any]):
        self.id: str = id
        self.description: str = desc
        self.config: dict[str, Any] = config

        self.taps: list[TappableAsync[Ctx]] = []

    def register(self, tappable: TappableAsync[Ctx]) -> None:
        self.taps.append(tappable)

    def unregister(self, tappable: TappableAsync[Ctx]) -> None:
        self.taps.remove(tappable)

    async def tap(self, ctx: Ctx) -> Ctx:
        for tappable in self.taps:
            ctx = await tappable.tap(ctx)

        return ctx


class Taps:
    def __init__(self) -> None:
        # self.tap_before_publish_inbound_msg: Tap[SomeContext] = Tap(
        #     "before_publish_inbound_msg", "Before publish inbound message", {}
        # )
        # self.tap_after_get_inbound_msg: Tap[SomeContext] = Tap(
        #     "after_get_inbound_msg", "After get inbound message", {}
        # )
        self.tap_async_consolidate_memory: TapAsync[TapConsolidateMemoryContext] = (
            TapAsync("consolidate_memory", "Consolidate memory", {})
        )


nanobot_taps = Taps()
