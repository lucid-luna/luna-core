# services/mcp/internal_server.py

from __future__ import annotations
from typing import Any
from contextlib import asynccontextmanager
from mcp.server.fastmcp import FastMCP, Context
from fastapi import FastAPI

class LunaMCPInternal:
    def __init__(self, *, name: str = "LUNA-MCP", version: str = "0.2.0",
                asr=None, emotion=None, multi_intent=None, vision=None,
                tts=None, translator=None, llm=None, memory=None, logger=None):
        self.logger = logger
        try:
            self.mcp = FastMCP(name, version=version, lifespan=self._lifespan)
        except TypeError:
            self.mcp = FastMCP(name, lifespan=self._lifespan)
        self._bind_resources(memory)
        self._bind_tools(translator, tts)

    @asynccontextmanager
    async def _lifespan(self, _server: FastMCP):
        yield

    def _bind_resources(self, memory):
        @self.mcp.resource("luna://health")
        def health() -> str:
            return "ok"

        if memory:
            @self.mcp.resource("luna://memory/stats")
            def memory_stats() -> str:
                stats = memory.get_memory_stats()
                return str(stats)

    def _bind_tools(self, translator, tts):
        if translator:
            @self.mcp.tool()
            def translate(text: str, from_lang: str, to_lang: str) -> str:
                return translator.translate(text=text, from_lang=from_lang, to_lang=to_lang)

        if tts:
            @self.mcp.tool()
            async def synthesize(text: str) -> str:
                result = await tts.synthesize_async(text=text, style=None, style_weight=None)
                return result.get("audio_url", "")

    def mount_sse(self, app: FastAPI, path: str = "/mcp"):
        app.mount(path, self.mcp.sse_app())
