# services/mcp/external_manager.py

from __future__ import annotations
import os
import traceback
import asyncio
import json
from pathlib import Path
from typing import Dict, Optional, List

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

from .types import MCPServerConfig

class ExternalMCPManager:
    def __init__(self, config_path: str, logger=None):
        self.config_path = Path(config_path)
        self.logger = logger
        self.configs: Dict[str, MCPServerConfig] = {}
        self.sessions: Dict[str, ClientSession] = {}
        self._client_tasks: Dict[str, asyncio.Task] = {}

    def load_config(self):
        data = {}
        if self.config_path.exists():
            data = json.loads(self.config_path.read_text(encoding="utf-8"))
        
        mcp_servers = data.get("mcpServers", {})
        
        if "servers" in data and not mcp_servers:
            servers = data.get("servers", [])
            mcp_servers = {s["id"]: s for s in servers}
        
        self.configs = {sid: MCPServerConfig(**cfg) for sid, cfg in mcp_servers.items()}
        
    
    def list_configs(self):
        return list(self.configs.values())

    async def start_enabled(self):
        self.load_config()
        for server_id, cfg in self.configs.items():
            if not cfg.enabled:
                continue
            if cfg.transport != "stdio":
                continue
            await self._start_stdio(server_id, cfg)
            
    async def reload_and_apply(self):
        self.load_config()
        for sid in list(self.sessions.keys()):
            cfg = self.configs.get(sid)
            if not cfg or not cfg.enabled or cfg.transport != "stdio":
                await self._stop_session(sid)
        for sid, cfg in self.configs.items():
            if cfg.enabled and sid not in self.sessions and cfg.transport == "stdio":
                await self._start_stdio(sid, cfg)

    async def _start_stdio(self, server_id: str, cfg: MCPServerConfig):
        env_vars = os.environ.copy()
        if cfg.env:
            env_vars.update(cfg.env)
            
        params = StdioServerParameters(
            command=cfg.command or "",
            args=cfg.args or [],
            env=env_vars,
            cwd=cfg.cwd or None,
        )
        self.logger.info(f"[L.U.N.A. ExternalMCPManager] 클라이언트 시작: {server_id}")
        self.logger.debug(f"[L.U.N.A. ExternalMCPManager] command={cfg.command}, args={cfg.args}, cwd={cfg.cwd}")
        self._client_tasks[server_id] = asyncio.create_task(self._run_client(server_id, params, cfg))
        
    async def _stop_session(self, server_id: str):
        task = self._client_tasks.pop(server_id, None)
        if task:
            task.cancel()
        self.sessions.pop(server_id, None)

    async def _run_client(self, server_id: str, params: StdioServerParameters, cfg: MCPServerConfig = None):
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"[L.U.N.A. ExternalMCPManager] '{server_id}' STDIO 클라이언트 연결 중... (시도 {attempt + 1}/{max_retries})")
                async with stdio_client(params) as (read, write):
                    self.logger.info(f"[L.U.N.A. ExternalMCPManager] '{server_id}' STDIO 스트림 열림. 세션 초기화 중...")
                    async with ClientSession(read, write) as session:
                        timeout_sec = (cfg.timeoutMs / 1000.0) if cfg else 30.0
                        try:
                            await asyncio.wait_for(session.initialize(), timeout=timeout_sec)
                        except asyncio.TimeoutError:
                            self.logger.error(f"[L.U.N.A. ExternalMCPManager] '{server_id}' 세션 초기화 타임아웃 ({timeout_sec}초)")
                            raise
                        self.logger.info(f"[L.U.N.A. ExternalMCPManager] '{server_id}' 세션 초기화 완료")
                        self.sessions[server_id] = session
                        await self._hold_until_closed(session)
                return
                
            except asyncio.CancelledError:
                self.logger.info(f"[L.U.N.A. ExternalMCPManager] '{server_id}' 작업 취소됨")
                return
            except Exception as e:
                self.logger.error(f"========== [L.U.N.A. Error Trace] ==========")
                self.logger.error(f"서버 ID: {server_id}")
                
                if hasattr(e, 'exceptions'):
                    for i, sub_exc in enumerate(e.exceptions):
                        self.logger.error(f"내부 에러 #{i+1}: {type(sub_exc).__name__} - {str(sub_exc)}")
                        tb_str = "".join(traceback.format_exception(type(sub_exc), sub_exc, sub_exc.__traceback__))
                        self.logger.error(f"스택 트레이스:\n{tb_str}")
                else:
                    tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
                    self.logger.error(f"상세 에러:\n{tb_str}")
                
                self.logger.error(f"============================================")
                        
                if attempt + 1 < max_retries:
                    wait_sec = 2 ** (attempt + 1)
                    self.logger.warning(f"[L.U.N.A. ExternalMCPManager] '{server_id}' 연결 실패 ({type(e).__name__}). {wait_sec}초 후 재시도...")
                    await asyncio.sleep(wait_sec)
                else:
                    if self.logger:
                        self.logger.error(f"[L.U.N.A. ExternalMCPManager] '{server_id}' 최대 재시도 횟수 초과. 연결 포기: {type(e).__name__}: {e}")
        
        self.sessions.pop(server_id, None)
        self.logger.info(f"[L.U.N.A. ExternalMCPManager] '{server_id}' 정리 완료")

    async def _hold_until_closed(self, session: ClientSession):
        """
        세션이 살아있는 동안 태스크를 유지
        """
        import asyncio

        try:
            while True:
                await asyncio.sleep(1)

                closed = False
                if hasattr(session, "closed"):
                    attr = session.closed
                    closed = attr() if callable(attr) else bool(attr)
                elif hasattr(session, "is_closed"):
                    attr = session.is_closed
                    closed = attr() if callable(attr) else bool(attr)

                if closed:
                    if self.logger:
                        self.logger.info(
                            "[L.U.N.A. ExternalMCPManager] 세션이 종료 신호를 보냈습니다."
                        )
                    break

        except asyncio.CancelledError:
            if self.logger:
                self.logger.info(
                    "[L.U.N.A. ExternalMCPManager] 세션 유지 태스크 취소됨"
                )
            return

    async def stop_all(self):
        for task in list(self._client_tasks.values()):
            task.cancel()
        self._client_tasks.clear()
        self.sessions.clear()
        
    async def start(self, server_id: str):
        cfg = self.configs.get(server_id)
        if not cfg:
            raise RuntimeError(f"'{server_id}' 설정 없음")
        if cfg.transport != "stdio":
            raise RuntimeError(f"'{server_id}' transport '{cfg.transport}' 지원 안 함")
        if server_id in self.sessions:
            return
        await self._start_stdio(server_id, cfg)
        
    async def stop(self, server_id: str):
        if server_id not in self.sessions and server_id not in self._client_tasks:
            return
        await self._stop_session(server_id)

    async def list_tools(self, server_id: str) -> List[types.Tool]:
        session = self.sessions.get(server_id)
        if not session:
            raise RuntimeError(f"'{server_id}' 세션 없음/비활성화")
        result = await session.list_tools()
        return result.tools if hasattr(result, 'tools') else result

    async def list_resources(self, server_id: str) -> List[types.Resource]:
        session = self.sessions.get(server_id)
        if not session:
            raise RuntimeError(f"'{server_id}' 세션 없음/비활성화")
        result = await session.list_resources()
        return result.resources if hasattr(result, 'resources') else result

    async def call_tool(self, server_id: str, tool_name: str, arguments: dict, timeout: float | None = None):
        import time
        call_start = time.time()
        
        session = self.sessions.get(server_id)
        if not session:
            self.logger.warning(f"[L.U.N.A. ExternalMCPManager] '{server_id}' 세션 없음 - 재연결 필요")
            raise RuntimeError(f"'{server_id}' 세션 없음/비활성화")
        
        self.logger.info(f"[L.U.N.A. ExternalMCPManager] '{server_id}/{tool_name}' 호출 시작 (세션 있음)")
        
        async def _invoke():
            return await session.call_tool(tool_name, arguments)
        
        try:
            if timeout and timeout > 0:
                result = await asyncio.wait_for(_invoke(), timeout=timeout)
            else:
                result = await _invoke()
            
            elapsed = time.time() - call_start
            self.logger.info(f"[L.U.N.A. ExternalMCPManager] '{server_id}/{tool_name}' 완료: {elapsed:.2f}s")
            return result
            
        except asyncio.TimeoutError:
            elapsed = time.time() - call_start
            self.logger.error(f"[L.U.N.A. ExternalMCPManager] '{server_id}/{tool_name}' 타임아웃: {elapsed:.2f}s")
            raise
