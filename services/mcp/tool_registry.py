# ====================================================================
#  File: services/mcp/tool_registry.py
# ====================================================================

from typing import List, Dict, Any
from mcp import types
from .spotify import SpotifyService

class ToolRegistry:
    def __init__(self, spotify_service: SpotifyService):
        """
        도구를 제공하는 모든 서비스를 등록하고 관리합니다.        
        """
        self.services = {
            "Spotify": spotify_service,
        }
        
    def get_all_tool_definitions(self) -> List[types.Tool]:
        """
        프로젝트의 모든 서비스에서 제공하는 도구의 Pydantic 객체를 수집합니다.
        """
        all_tools = []
        for service in self.services.values():
            if hasattr(service, 'get_tool_definitions'):
                all_tools.extend(service.get_tool_definitions())
        return all_tools
    
    def dispatch_tool_call(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        도구 호출 요청을 적절한 서비스로 전달합니다.
        
        Args:
            name (str): 호출할 도구의 이름
            arguments (Dict[str, Any]): 도구에 전달할 인수
        Returns:
            Any: 도구 실행 결과
        """
        service_name = name.replace("Playback", "").replace("Search", "")
        
        if service_name in self.services:
            service = self.services[service_name]
            return service.call_tool(name, arguments)
        else:
            raise ValueError(f"[L.U.N.A. MCP] 알 수 없는 도구 {name}을(를) 호출하려고 시도했습니다.")