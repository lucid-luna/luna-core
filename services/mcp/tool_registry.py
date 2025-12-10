# ====================================================================
#  File: services/mcp/tool_registry.py
# ====================================================================

from __future__ import annotations
from typing import Callable, Dict, Any, Optional
import inspect

class ToolRegistry:
    """
    L.U.N.A. 코어 공용 툴 레지스트리
    """
    def __init__(self, **kwargs):
        self._tools: Dict[str, Callable[..., Any]] = {}

    # -------------------- 등록 / 해제 --------------------

    def register(self, name: str, func: Callable[..., Any]) -> None:
        """
        도구 등록

        Args:
            name: 고유 이름 (예: "notion/createPage")
            func: 호출 가능한 함수 (sync 또는 async)
        """
        if not callable(func):
            raise TypeError(f"Tool '{name}' must be callable")
        self._tools[name] = func

    def unregister(self, name: str) -> None:
        """등록된 도구 제거 (없으면 조용히 무시)"""
        self._tools.pop(name, None)

    # -------------------- 조회 유틸 --------------------

    def list(self) -> list[str]:
        """등록된 도구 이름 목록"""
        return list(self._tools.keys())

    def has(self, name: str) -> bool:
        """해당 이름의 도구 존재 여부"""
        return name in self._tools

    def get(self, name: str) -> Optional[Callable[..., Any]]:
        """도구 함수 직접 가져오기 (없으면 None)"""
        return self._tools.get(name)

    # -------------------- 실행 (sync / async) --------------------

    def call(self, name: str, **kwargs) -> Any:
        """
        동기 방식 호출.

        - 순수 sync 도구에만 권장.
        - async 도구를 등록해두었다면, 이 메서드 대신 call_async 를 사용해야 한다.
        """
        if name not in self._tools:
            raise KeyError(f"Tool not found: {name}")

        func = self._tools[name]
        result = func(**kwargs)

        # 실수로 async 함수를 call()로 부른 경우 방지용
        if inspect.isawaitable(result):
            raise RuntimeError(
                f"Tool '{name}' is async; use 'await ToolRegistry.call_async(...)' instead."
            )
        return result

    async def call_async(self, name: str, **kwargs) -> Any:
        """
        비동기 방식 호출.

        - sync / async 도구 모두 지원.
        - InteractionService 같은 비동기 플로우에서는 이 메서드를 사용하는 게 안전하다.
        """
        if name not in self._tools:
            raise KeyError(f"Tool not found: {name}")

        func = self._tools[name]
        result = func(**kwargs)

        if inspect.isawaitable(result):
            return await result
        return result