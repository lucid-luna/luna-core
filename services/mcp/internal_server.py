# ====================================================================
#  File: services/mcp/internal_server.py
# ====================================================================

from __future__ import annotations
from typing import Any, Optional
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP, Context
from fastapi import FastAPI


class LunaMCPInternal:
    """
    L.U.N.A. 내부 MCP 서버

    - 기본 제공 리소스:
        - luna://health
        - luna://memory/stats (memory 서비스가 있을 때)

    - 선택적 도구:
        - translate(...) / synthesize(...) 는 기본 비활성화.
            → 파이프라인에서 직접 translator/tts를 쓰기 때문에
                LLM 쪽 MCP 도구 목록에는 굳이 안 노출하는 게 안전함.
    """

    def __init__(
        self,
        *,
        name: str = "LUNA-MCP",
        version: str = "0.2.0",
        asr: Any = None,
        emotion: Any = None,
        multi_intent: Any = None,
        vision: Any = None,
        tts: Any = None,
        translator: Any = None,
        llm: Any = None,
        memory: Any = None,
        logger: Optional[Any] = None,
        expose_translator_tool: bool = False,
        expose_tts_tool: bool = False,
    ):
        self.logger = logger

        try:
            self.mcp = FastMCP(name, version=version, lifespan=self._lifespan)
        except TypeError:
            # fastmcp 버전에 따라 version 인자가 없을 수 있음
            self.mcp = FastMCP(name, lifespan=self._lifespan)

        # 리소스 바인딩
        self._bind_resources(memory)

        # 도구 바인딩 (옵션)
        self._bind_tools(
            translator if expose_translator_tool else None,
            tts if expose_tts_tool else None,
        )

    @asynccontextmanager
    async def _lifespan(self, _server: FastMCP):
        # 필요하면 여기서 init/cleanup 로직 추가 가능
        yield

    # ------------------------------------------------------------------
    # 리소스
    # ------------------------------------------------------------------

    def _bind_resources(self, memory):
        @self.mcp.resource("luna://health")
        def health() -> str:
            return "ok"

        if memory:
            @self.mcp.resource("luna://memory/stats")
            def memory_stats() -> str:
                stats = memory.get_memory_stats()
                # 나중에 원하면 JSON으로 바꿔도 됨
                return str(stats)

    # ------------------------------------------------------------------
    # 도구 (옵션)
    # ------------------------------------------------------------------

    def _bind_tools(self, translator, tts):
        # 기본값: 노출 안 함
        if translator:
            @self.mcp.tool()
            def translate(text: str, from_lang: str, to_lang: str) -> str:
                return translator.translate(
                    text=text,
                    from_lang=from_lang,
                    to_lang=to_lang,
                )

        if tts:
            @self.mcp.tool()
            async def synthesize(text: str) -> str:
                result = await tts.synthesize_async(
                    text=text,
                    style=None,
                    style_weight=None,
                )
                return result.get("audio_url", "")

    # ------------------------------------------------------------------
    # FastAPI 마운트
    # ------------------------------------------------------------------

    def mount_sse(self, app: FastAPI, path: str = "/mcp"):
        """
        내부 MCP 서버를 FastAPI 앱에 SSE 엔드포인트로 마운트.
        """
        app.mount(path, self.mcp.sse_app())
