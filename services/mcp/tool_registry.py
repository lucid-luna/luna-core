# ====================================================================
#  File: services/mcp/tool_registry.py
# ====================================================================

from __future__ import annotations
from typing import Callable, Dict, Any, Optional

class ToolRegistry:
    """
    L.U.N.A. 코어 공용 툴 레지스트리
    """
    def __init__(self, **kwargs):
        self._tools: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str, func: Callable[..., Any]) -> None:
        if not callable(func):
            raise TypeError(f"Tool '{name}' must be callable")
        self._tools[name] = func

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def list(self) -> list[str]:
        return list(self._tools.keys())

    def has(self, name: str) -> bool:
        return name in self._tools

    def call(self, name: str, **kwargs) -> Any:
        if name not in self._tools:
            raise KeyError(f"Tool not found: {name}")
        return self._tools[name](**kwargs)