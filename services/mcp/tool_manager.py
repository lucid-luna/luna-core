# services/mcp/tool_manager.py

from __future__ import annotations
import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable

from mcp import types

from .external_manager import ExternalMCPManager
from .tool_registry import ToolRegistry

class MCPToolManager:
    """
    외부 MCP 서버에서 도구 목록을 가져와 ToolRegistry에 등록하고,
    실제 호출을 중계하는 매니저.
    """

    def __init__(
        self,
        external_manager: ExternalMCPManager,
        tool_registry: ToolRegistry,
        logger: Optional[logging.Logger] = None,
    ):
        self.external_manager = external_manager
        self.tool_registry = tool_registry
        self.logger = logger or logging.getLogger(__name__)

        # server_id -> {tool_name -> Tool}
        self._tool_map: Dict[str, Dict[str, types.Tool]] = {}

        # server_id -> namespace (없으면 server_id 자체를 namespace로)
        self._namespace_map: Dict[str, Optional[str]] = {}

    # ------------------------------------------------------------------
    # 초기화 / 재로드
    # ------------------------------------------------------------------

    async def initialize(self):
        """
        활성화된 MCP 서버들의 도구를 모두 수집하고 ToolRegistry에 등록.
        앱 부팅 시 한 번 호출.
        """
        self.logger.info("[L.U.N.A. MCPToolManager] 초기화 시작...")

        for config in self.external_manager.list_configs():
            if not config.enabled:
                continue

            server_id = config.id

            self.logger.info(f"[L.U.N.A. MCPToolManager] {server_id} 도구 수집 시작...")
            max_wait = 300  # 300 * 0.1s = 30초
            waited = 0

            import time
            start_time = time.time()

            # ExternalMCPManager.start_enabled() 가 세션을 열 때까지 대기
            while server_id not in self.external_manager.sessions and waited < max_wait:
                await asyncio.sleep(0.1)
                waited += 1

            elapsed = time.time() - start_time

            if server_id not in self.external_manager.sessions:
                self.logger.warning(
                    f"[L.U.N.A. MCPToolManager] {server_id} 세션 준비 실패 "
                    f"(대기: {elapsed:.1f}초) - 나중에 재시도"
                )
                continue

            self.logger.debug(
                f"[L.U.N.A. MCPToolManager] {server_id} 세션 준비 완료 "
                f"(소요: {elapsed:.1f}초)"
            )

            # 실제 도구 목록 조회
            try:
                tools = await asyncio.wait_for(
                    self.external_manager.list_tools(server_id),
                    timeout=10.0,
                )
                self.logger.info(
                    f"[L.U.N.A. MCPToolManager] {server_id}: {len(tools)}개 도구 발견"
                )

                self._tool_map[server_id] = {tool.name: tool for tool in tools}

                if config.namespace:
                    self._namespace_map[server_id] = config.namespace

                # ToolRegistry에 namespaced 이름으로 등록
                for tool in tools:
                    tool_name = self._get_namespaced_tool_name(server_id, tool.name)

                    async def call_tool_wrapper(
                        arguments: dict,
                        _server_id: str = server_id,
                        _tool_name: str = tool.name,
                    ) -> Any:
                        return await self.call_tool(_server_id, _tool_name, arguments)

                    self.tool_registry.register(tool_name, call_tool_wrapper)
                    self.logger.debug(
                        f"[L.U.N.A. MCPToolManager] 도구 등록: {tool_name}"
                    )

            except asyncio.TimeoutError:
                self.logger.warning(
                    f"[L.U.N.A. MCPToolManager] {server_id} 도구 목록 조회 타임아웃 (10초)"
                )
            except Exception as e:
                self.logger.error(
                    f"[L.U.N.A. MCPToolManager] {server_id} 도구 수집 실패: {e}"
                )

        self.logger.info(
            f"[L.U.N.A. MCPToolManager] 초기화 완료: "
            f"총 {len(self._tool_map)}개 서버, "
            f"총 {sum(len(tools) for tools in self._tool_map.values())}개 도구"
        )

        if self._tool_map:
            for server_id, tools in self._tool_map.items():
                tool_names = ", ".join(tools.keys())
                self.logger.info(
                    f"[L.U.N.A. MCPToolManager] {server_id} 도구: {tool_names}"
                )

    async def reload(self):
        """
        도구 목록 재로드.

        - 기존에 등록된 MCP 도구들을 ToolRegistry에서 제거 후
        - 다시 initialize() 수행
        """
        self.logger.info("[L.U.N.A. MCPToolManager] 도구 목록 재로드 시작...")

        for server_id in list(self._tool_map.keys()):
            for tool_name in list(self._tool_map[server_id].keys()):
                full_name = self._get_namespaced_tool_name(server_id, tool_name)
                self.tool_registry.unregister(full_name)

        self._tool_map.clear()
        self._namespace_map.clear()

        await self.initialize()

    # ------------------------------------------------------------------
    # 호출
    # ------------------------------------------------------------------

    async def call_tool(
        self,
        server_id: str,
        tool_name: str,
        arguments: dict,
        timeout: Optional[float] = 8.0,
    ) -> Any:
        """
        실제 MCP 서버의 도구를 호출.

        - InteractionService → ToolRegistry.call_async → 여기로 내려오는 플로우.
        """
        try:
            self.logger.info(
                f"[L.U.N.A. MCPToolManager] 도구 호출: {server_id}/{tool_name} "
                f"(args: {list(arguments.keys())})"
            )

            result = await self.external_manager.call_tool(
                server_id,
                tool_name,
                arguments,
                timeout=timeout,
            )

            # result 는 MCP 프로토콜의 ToolResult 구조 그대로 올 수 있음
            self.logger.debug(
                f"[L.U.N.A. MCPToolManager] 도구 호출 성공: {str(result)[:500]}"
            )
            return result

        except asyncio.TimeoutError:
            self.logger.error(
                f"[L.U.N.A. MCPToolManager] {server_id}/{tool_name} "
                f"도구 호출 타임아웃 ({timeout}초)"
            )
            raise

        except Exception as e:
            self.logger.error(f"[L.U.N.A. MCPToolManager] 도구 호출 실패: {e}")
            raise

    # ------------------------------------------------------------------
    # LLM용 도구 스펙 제공
    # ------------------------------------------------------------------

    def get_tool_list(self) -> List[Dict[str, Any]]:
        """
        모든 등록된 도구의 "스펙" 목록을 반환.

        LLMManager.generate(...) 에 넘기는 tools 파라미터에 그대로 사용 가능.
        """
        tools: List[Dict[str, Any]] = []

        for server_id, tool_dict in self._tool_map.items():
            namespace = self._namespace_map.get(server_id, server_id)

            for tool_name, tool in tool_dict.items():
                schema = {}
                try:
                    schema = getattr(tool, "inputSchema", {}) or {}
                    if hasattr(schema, "model_dump"):
                        schema = schema.model_dump()
                    elif hasattr(schema, "dict"):
                        schema = schema.dict()
                except Exception:
                    schema = {}

                full_id = f"{server_id}/{tool_name}"
                display_name = f"{namespace}/{tool_name}"

                tools.append(
                    {
                        "id": full_id,
                        "name": display_name,
                        "description": getattr(tool, "description", "") or "",
                        "inputSchema": schema,
                        "server_id": server_id,
                        "server": server_id,
                        "function": {
                            "name": tool_name,
                            "description": getattr(tool, "description", "") or "",
                            "parameters": schema,
                            "inputSchema": schema,
                        },
                    }
                )

        return tools

    def get_tool_info(self, server_id: str, tool_name: str) -> Optional[types.Tool]:
        """원본 MCP Tool 객체 조회"""
        return self._tool_map.get(server_id, {}).get(tool_name)

    # ------------------------------------------------------------------
    # 리소스
    # ------------------------------------------------------------------

    async def list_resources(self, server_id: str) -> List[types.Resource]:
        try:
            resources = await self.external_manager.list_resources(server_id)
            self.logger.debug(
                f"[L.U.N.A. MCPToolManager] {server_id} 리소스: {len(resources)}개"
            )
            return resources
        except Exception as e:
            self.logger.error(
                f"[L.U.N.A. MCPToolManager] {server_id} 리소스 조회 실패: {e}"
            )
            return []

    # ------------------------------------------------------------------
    # 내부 유틸
    # ------------------------------------------------------------------

    def _get_namespaced_tool_name(self, server_id: str, tool_name: str) -> str:
        """
        LLM / 레지스트리에서 사용하는 full name 생성.

        예: server_id="youtube-music", namespace="music", tool_name="playTrack"
            -> "music/playTrack"
        """
        namespace = self._namespace_map.get(server_id, server_id)
        return f"{namespace}/{tool_name}"