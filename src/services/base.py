from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable


@runtime_checkable
class ServiceProtocol(Protocol):
    async def initialize(self) -> None: ...
    async def shutdown(self) -> None: ...


class BaseService(ABC):
    def __init__(self) -> None:
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        pass

    @property
    def is_initialized(self) -> bool:
        return self._initialized
