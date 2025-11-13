# services/mcp/tool_manager.py

from __future__ import annotations
import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from mcp import types

from .external_manager import ExternalMCPManager
from .tool_registry import ToolRegistry


class MCPToolManager:
    def __init__(
        self, 
        external_manager: ExternalMCPManager,
        tool_registry: ToolRegistry,
        logger: Optional[logging.Logger] = None
    ):
        self.external_manager = external_manager
        self.tool_registry = tool_registry
        self.logger = logger or logging.getLogger(__name__)
        
        self._tool_map: Dict[str, Dict[str, types.Tool]] = {}
        
        self._namespace_map: Dict[str, Optional[str]] = {}

    async def initialize(self):
        self.logger.info("[L.U.N.A. MCPToolManager] 초기화 시작...")
        
        for config in self.external_manager.list_configs():
            if not config.enabled:
                continue
                
            server_id = config.id
            
            self.logger.info(f"[L.U.N.A. MCPToolManager] {server_id} 도구 수집 시작...")
            max_wait = 300
            waited = 0
            import time
            start_time = time.time()
            while server_id not in self.external_manager.sessions and waited < max_wait:
                await asyncio.sleep(0.1)
                waited += 1
            elapsed = time.time() - start_time
            
            if server_id not in self.external_manager.sessions:
                self.logger.warning(f"[L.U.N.A. CPToolManager] {server_id} 세션 준비 실패 (대기: {elapsed:.1f}초) - 나중에 재시도")
                continue
            
            self.logger.debug(f"[L.U.N.A. MCPToolManager] {server_id} 세션 준비 완료 (소요: {elapsed:.1f}초)")
            
            try:
                tools = await asyncio.wait_for(
                    self.external_manager.list_tools(server_id),
                    timeout=10.0
                )
                self.logger.info(f"[L.U.N.A. MCPToolManager] {server_id}: {len(tools)}개 도구 발견")
                
                self._tool_map[server_id] = {tool.name: tool for tool in tools}
                
                if config.namespace:
                    self._namespace_map[server_id] = config.namespace
                
                for tool in tools:
                    tool_name = self._get_namespaced_tool_name(server_id, tool.name)
                    
                    async def call_tool_wrapper(
                        arguments: dict,
                        _server_id: str = server_id,
                        _tool_name: str = tool.name
                    ) -> Any:
                        return await self.call_tool(_server_id, _tool_name, arguments)
                    
                    self.tool_registry.register(tool_name, call_tool_wrapper)
                    self.logger.debug(f"[L.U.N.A. MCPToolManager] 도구 등록: {tool_name}")
                    
            except asyncio.TimeoutError:
                self.logger.warning(f"[L.U.N.A. MCPToolManager] {server_id} 도구 목록 조회 타임아웃 (10초) - 서버가 응답하지 않음")
            except Exception as e:
                self.logger.error(f"[L.U.N.A. MCPToolManager] {server_id} 도구 수집 실패: {e}")
        
        self.logger.info(
            f"[L.U.N.A. MCPToolManager] 초기화 완료: "
            f"총 {len(self._tool_map)}개 서버, "
            f"총 {sum(len(tools) for tools in self._tool_map.values())}개 도구"
        )
        
        if self._tool_map:
            for server_id, tools in self._tool_map.items():
                tool_names = ", ".join(tools.keys())
                self.logger.info(f"[L.U.N.A. MCPToolManager] {server_id} 도구: {tool_names}")

    async def reload(self):
        self.logger.info("[L.U.N.A. MCPToolManager] 도구 목록 재로드 시작...")
        
        for server_id in list(self._tool_map.keys()):
            for tool_name in list(self._tool_map[server_id].keys()):
                full_name = self._get_namespaced_tool_name(server_id, tool_name)
                self.tool_registry.unregister(full_name)
        
        self._tool_map.clear()
        self._namespace_map.clear()
        
        await self.initialize()

    async def call_tool(
        self, 
        server_id: str, 
        tool_name: str, 
        arguments: dict,
        timeout: float | None = 8.0,
    ) -> Any:
        try:
            self.logger.info(
                f"[L.U.N.A. MCPToolManager] 도구 호출: {server_id}/{tool_name} "
                f"(args: {list(arguments.keys())})"
            )
            
            result = await self.external_manager.call_tool(
                server_id, 
                tool_name, 
                arguments,
                timeout=timeout
            )
            
            self.logger.debug(f"[L.U.N.A. MCPToolManager] 도구 호출 성공: {str(result)[:500]}")
            return result
        
        except asyncio.TimeoutError:
            self.logger.error(f"[L.U.N.A. MCPToolManager] {server_id}/{tool_name} 도구 호출 타임아웃 ({timeout}초)")
            raise
            
        except Exception as e:
            self.logger.error(f"[MCPToolManager] 도구 호출 실패: {e}")
            raise

    def get_tool_list(self) -> List[Dict[str, Any]]:
        """
        모든 등록된 도구의 목록을 반환합니다.

        반환 형식(예):
        {
        "id": "youtube-music/playTrack",
        "server_id": "youtube-music",
        "server": "youtube-music",
        "name": "youtube-music/playTrack",
        "description": "...",
        "inputSchema": {...},
        "function": {
            "name": "playTrack",
            "description": "...",
            "parameters": {...},
            "inputSchema": {...}
        }
        }
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

                tools.append({
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
                })

        return tools

    def get_tool_info(self, server_id: str, tool_name: str) -> Optional[types.Tool]:
        return self._tool_map.get(server_id, {}).get(tool_name)

    def _get_namespaced_tool_name(self, server_id: str, tool_name: str) -> str:
        namespace = self._namespace_map.get(server_id, server_id)
        return f"{namespace}/{tool_name}"

    async def list_resources(self, server_id: str) -> List[types.Resource]:
        try:
            resources = await self.external_manager.list_resources(server_id)
            self.logger.debug(f"[L.U.N.A. MCPToolManager] {server_id} 리소스: {len(resources)}개")
            return resources
        except Exception as e:
            self.logger.error(f"[L.U.N.A. MCPToolManager] {server_id} 리소스 조회 실패: {e}")
            return []
