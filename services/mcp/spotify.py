# ====================================================================
#  File: services/mcp/spotify.py
# ====================================================================

import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from .spotify_api import SpotifyAPIClient

class Playback(BaseModel):
    """스포티파이 재생을 관리합니다. 사용 가능한 동작: 'get'(현재 곡 정보), 'start'(재생/재개), 'pause'(일시정지), 'skip'(다음 곡)."""
    action: str = Field(description="'get', 'start', 'pause', 'skip' 중 하나의 동작")
    spotify_uri: Optional[str] = Field(default=None, description="'start' 동작 시 재생할 특정 트랙/앨범/플레이리스트의 URI")

class Search(BaseModel):
    """스포티파이에서 트랙, 앨범, 아티스트를 검색합니다. 검색 결과를 바탕으로 재생할 수 있습니다."""
    query: str = Field(description="검색어 (예: '아이유 밤편지')")
    qtype: Optional[str] = Field(default="track", description="검색 타입 (예: 'track', 'album', 'artist')")
    limit: Optional[int] = Field(default=5, description="반환할 최대 결과 수")

class SpotifyService:
    def __init__(self, logger):
        """SpotifyService를 초기화하고 실제 API 클라이언트를 생성합니다."""
        self.spotify_client = SpotifyAPIClient(logger)
        self.logger = logger
        self.tool_executors = {
            "SpotifyPlayback": self.execute_playback,
            "SpotifySearch": self.execute_search,
        }

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """LLM 에이전트에게 제공할 모든 스포티파이 Tool의 공식 정의를 반환합니다."""
        if not self.spotify_client.sp:
            self.logger.warning("[SpotifyService] 인증 실패로 스포티파이 도구를 등록하지 않습니다.")
            return []

        return [
            {
                "type": "function",
                "function": {
                    "name": "SpotifyPlayback",
                    "description": Playback.__doc__,
                    "parameters": Playback.model_json_schema()
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "SpotifySearch",
                    "description": Search.__doc__,
                    "parameters": Search.model_json_schema()
                }
            }
        ]

    def call_tool(self, name: str, arguments: dict) -> Any:
        """LLM의 Tool 호출 요청을 받아 적절한 실행 메서드로 전달합니다."""
        if name in self.tool_executors:
            return self.tool_executors[name](**arguments)
        else:
            raise ValueError(f"알 수 없는 스포티파이 도구입니다: {name}")

    def execute_playback(self, action: str, spotify_uri: Optional[str] = None) -> str:
        """재생 관련 Tool의 실제 로직을 수행합니다."""
        self.logger.info(f"Executing Playback: action={action}, uri={spotify_uri}")
        if not self.spotify_client.sp:
            return "스포티파이 인증에 실패하여 실행할 수 없습니다."

        # 중요: 실제 명령을 내릴 기기(Device)의 ID를 먼저 가져옵니다.
        device_id = self.spotify_client.get_active_device()
        if not device_id and action in ["start", "pause", "skip"]:
            return "명령을 실행할 스포티파이 기기(PC, 스마트폰 앱 등)를 찾을 수 없습니다. 스포티파이가 켜져 있는지 확인해주세요."

        match action:
            case "get":
                track_info = self.spotify_client.get_current_track()
                if track_info and track_info.get('item'):
                    item = track_info['item']
                    artist = item['artists'][0]['name']
                    title = item['name']
                    return f"현재 재생 중인 곡은 '{artist}'의 '{title}'입니다."
                return "현재 재생 중인 트랙이 없습니다."
            case "start":
                self.spotify_client.start_playback(spotify_uri=spotify_uri, device_id=device_id)
                return "재생을 시작합니다."
            case "pause":
                self.spotify_client.pause_playback(device_id=device_id)
                return "재생을 일시정지합니다."
            case "skip":
                self.spotify_client.skip_track(device_id=device_id)
                return "다음 곡으로 건너뜁니다."
            case _:
                return f"알 수 없는 재생 동작입니다: {action}"

    def execute_search(self, query: str, qtype: str = "track", limit: int = 5) -> str:
        """검색 Tool의 실제 로직을 수행합니다."""
        self.logger.info(f"Executing Search: query={query}, type={qtype}")
        if not self.spotify_client.sp:
            return "스포티파이 인증에 실패하여 실행할 수 없습니다."

        results = self.spotify_client.search(query=query, qtype=qtype, limit=limit)
        
        if results and results.get('tracks') and results['tracks'].get('items'):
            # LLM이 검색 결과를 보고 다음 행동(예: 재생)을 결정하기 쉽도록 가공합니다.
            formatted_results = [
                {
                    "artist": item['artists'][0]['name'],
                    "title": item['name'],
                    "uri": item['uri']  # 재생에 필요한 고유 URI
                }
                for item in results['tracks']['items']
            ]
            # JSON 문자열로 변환하여 반환
            return json.dumps(formatted_results, ensure_ascii=False)
        
        return f"'{query}'에 대한 검색 결과가 없습니다."