# services/mcp/types.py

from __future__ import annotations
from typing import Optional, Dict, List, Literal
from pydantic import BaseModel, Field

Transport = Literal["stdio", "sse"]

class MCPServerConfig(BaseModel):
    id: str = Field(..., description="고유 ID")
    transport: Transport = "stdio"
    command: Optional[str] = None
    args: Optional[List[str]] = None
    cwd: Optional[str] = None
    env: Optional[Dict[str, str]] = None
    enabled: bool = False
    namespace: Optional[str] = None
    timeoutMs: Optional[int] = 8000