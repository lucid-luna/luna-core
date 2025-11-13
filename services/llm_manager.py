# ====================================================================
#  File: services/llm_manager.py
# ====================================================================

from typing import Dict, List, Any, Optional, Literal
from services.llm import LLMService
from services.llm_api import LLMAPIService


class LLMManager:
    def __init__(
        self,
        mode: Literal["server", "api"] = "server",
        server_configs: Optional[Dict[str, Dict[str, str]]] = None,
        api_configs: Optional[Dict[str, Dict[str, str]]] = None
    ):
        """
        LLM 관리자 - 로컬 서버 또는 API 방식을 선택적으로 사용

        Args:
            mode (str): "server" (로컬 LLM 서버) 또는 "api" (Gemini 등의 API)
            server_configs (Dict, optional): LLM 서버 설정
                예: {
                    "luna": {
                        "url": "http://localhost:8080",
                        "alias": "luna-model"
                    }
                }
            api_configs (Dict, optional): API 제공자 설정
                예: {
                    "gemini": {
                        "api_key": "your-api-key",
                        "model": "gemini-2.5-flash"
                    }
                }
        """
        self.mode = mode
        self.service = None
        
        if mode == "server":
            if server_configs is None:
                raise ValueError("[L.U.N.A. LLM Manager] 서버 모드에는 server_configs가 필요합니다.")
            self.service = LLMService(server_configs=server_configs)
            print(f"[L.U.N.A. LLM Manager] 로컬 서버 모드로 초기화 완료")
        
        elif mode == "api":
            if api_configs is None:
                raise ValueError("[L.U.N.A. LLM Manager] API 모드에는 api_configs가 필요합니다.")
            self.service = LLMAPIService(api_configs=api_configs)
            print(f"[L.U.N.A. LLM Manager] API 모드로 초기화 완료")
        
        else:
            raise ValueError(f"[L.U.N.A. LLM Manager] 지원하지 않는 모드입니다: {mode}")
    
    def generate(
        self,
        target: str,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Any]] = None,
        temperature: float = 0.9,
        max_tokens: int = 128,
        model: Optional[str] = None,
        stream: bool = False,
        skip_cache: bool = False
    ) -> Dict[str, Any]:
        """
        LLM에 프롬프트를 보내고 응답을 받습니다.

        Args:
            target (str): 
                - server 모드: 서버 이름 (예: "luna")
                - api 모드: API 제공자 (예: "gemini")
            system_prompt (str, optional): 시스템 프롬프트
            user_prompt (str, optional): 사용자 프롬프트
            messages (List[Dict[str, Any]], optional): 대화 내역
            tools (List[Any], optional): 사용할 도구 목록 (서버 모드만 지원)
            temperature (float, optional): 생성의 다양성 조절 (서버 모드만 지원)
            max_tokens (int, optional): 생성할 최대 토큰 수 (서버 모드만 지원)
            model (str, optional): 사용할 모델 (API 모드에서 사용)
            stream (bool): 스트리밍 모드 활성화 (API 모드만 지원)
            skip_cache (bool): 캐시를 무시하고 새로 생성 (도구 사용 시)

        Returns:
            Dict[str, Any]: LLM 응답 또는 스트림 제너레이터
        """
        if self.mode == "server":
            return self.service.generate(
                target_server=target,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                skip_cache=skip_cache
            )
        
        elif self.mode == "api":
            return self.service.generate(
                target_provider=target,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                messages=messages,
                tools=tools,  # ← MCP 도구 전달
                model=model,
                stream=stream,
                skip_cache=skip_cache
            )
        
        return {"error": "Invalid mode"}
    
    def get_mode(self) -> str:
        """현재 모드 반환"""
        return self.mode
    
    def get_available_targets(self) -> List[str]:
        """사용 가능한 타겟 목록 반환"""
        if self.mode == "server":
            return list(self.service.server_configs.keys())
        elif self.mode == "api":
            return self.service.get_available_providers()
        return []
