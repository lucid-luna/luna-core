# ====================================================================
#  File: services/mcp/spotify_api.py
# ====================================================================

import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv

load_dotenv()  # .env 파일에서 환경 변수 로드

class SpotifyAPIClient:
    def __init__(self, logger):
        self.logger = logger
        try:
            auth_manager = SpotifyOAuth(
                client_id=os.getenv("SPOTIFY_CLIENT_ID"),
                client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
                redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI"),
                scope="user-read-playback-state,user-modify-playback-state,user-read-currently-playing,playlist-modify-public,playlist-read-private",
                cache_path=".spotify_cache"
            )
            self.sp = spotipy.Spotify(auth_manager=auth_manager)
            
            # 초기 인증 토큰 확인
            self.sp.current_user()
            self.logger.info("[SpotifyAPIClient] Spotify API 클라이언트 초기화 완료.")
        except Exception as e:
            self.logger.error(f"[SpotifyAPIClient] 스포티파이 인증 실패: {e}")
            self.logger.error("환경변수(.env) 설정 및 Spotify 개발자 대시보드의 Redirect URI 설정을 확인하세요.")
            self.sp = None
            
    def get_current_track(self):
        if not self.sp: return None
        return self.sp.current_playback()
    
    def play(self, spotify_uri: str = None):
        if not self.sp: return
        uris = [spotify_uri] if spotify_uri else None
        self.sp.start_playback(uris=uris)
    
    def pause(self):
        if not self.sp: return
        self.sp.pause_playback()
    
    def skip(self):
        if not self.sp: return
        self.sp.next_track()
        
    def search(self, query: str, qtype: str, limit: int):
        if not self.sp: return None
        return self.sp.search(q=query, type=qtype, limit=limit)
    
    def get_active_device(self):
        if not self.sp: 
            return None
        
        playback_state = self.sp.current_playback()
        
        if playback_state and playback_state.get('device') and playback_state['device'].get('id'):
            return playback_state['device']['id']
        
        self.logger.warning("활성 스포티파이 기기를 찾을 수 없습니다. 스포티파이 앱/웹에서 음악을 한번이라도 재생했는지 확인해주세요.")
        return None
