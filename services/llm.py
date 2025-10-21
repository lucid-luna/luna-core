# ====================================================================
#  File: services/llm.py
# ====================================================================
\
import requests
from typing import Dict, List, Any, Optional

class LLMService:
    def __init__(self, server_configs: Dict[str, Dict[str, str]]):
        """
        Luna LLM 서비스 초기화

        Args:
            server_url (Dict[str, str]): LLM 서버의 URL
        """
        self.server_configs = server_configs
    
    def generate(
        self,
        target_server: str,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Any]] = None,
        temperature: float = 0.9,
        max_tokens: int = 128
    ) -> Dict[str, Any]:
        """
        LLM 서버에 프롬프트를 보내고 응답을 받습니다.

        Args:
            system_prompt (str): 시스템 프롬프트
            user_prompt (str): 사용자 프롬프트
            target_server (str): 요청을 보낼 LLM 서버
            messages (List[Dict[str, Any]], optional): 대화 내역
            tools (List[Any], optional): 사용할 도구 목록
            temperature (float, optional): 생성의 다양성 조절 파라미터
            max_tokens (int, optional): 생성할 최대 토큰 수
        Returns:
            Dict[str, Any]: LLM 서버의 응답
        """
        if target_server not in self.server_configs:
            print(f"[L.U.N.A. LLM] 알 수 없는 서버 타겟입니다: {target_server}")
            return ""

        config = self.server_configs[target_server]
        server_url = config["url"].rstrip('/')
        model_alias = config["alias"]

        if not messages:
            if system_prompt is None or user_prompt is None:
                raise ValueError("[L.U.N.A. LLM] 시스템 프롬프트와 사용자 프롬프트는 필수입니다.")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        
        payload = {
            "model": model_alias,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        try:
            response = requests.post(
                f"{server_url}/v1/chat/completions",
                json=payload,
                timeout=300
            )
            response.raise_for_status()
            return response.json()
        
        except requests.RequestException as e:
            print(f"[L.U.N.A. LLM] '{target_server}' 서버 요청 중 오류 발생: {e}")
            return {}