# ====================================================================
#  File: services/tts.py
# ====================================================================

from pathlib import Path
import requests

class LLMService:
    def __init__(self, server_url: str, prompt_path: str = "./checkpoints/LunaLLM/prompt.txt"):
        self.url = server_url.rstrip('/')
        self.prompt_template = self._load_prompt(prompt_path)
        
    def _load_prompt(self, prompt_path: str) -> str:
        """
        경로에서 프롬프트를 로드합니다.
        """
        try:
            return Path(prompt_path).read_text(encoding='utf-8')
        except Exception as e:
            print(f"[L.U.N.A. LLM] 프롬프트를 불러오는 중 오류가 발생했습니다: {e}")
            return "User: {input}\nAssistant:"
    
    def generate(self, input_text: str, temperature: float = 0.7, max_tokens: int = 256) -> str:
        """
        LLM 서버에 요청을 보내고 응답을 반환합니다.
        
        Args:
            input_text (str): 사용자 입력 텍스트
            temperature (float): 생성의 다양성 조절 파라미터
            max_tokens (int): 생성할 최대 토큰 수
            
        Returns:
            str: LLM의 응답 텍스트
        """
        messages = [
            {
                "role": "system",
                "content": self.prompt_template.strip()
            },
            {
                "role": "user",
                "content": input_text
            }
        ]
        
        payload = {
            "model": "Luna",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
                
        try:
            response = requests.post(f"{self.url}/v1/chat/completions", json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            return result["choices"][0]["message"]["content"].strip()
        except requests.RequestException as e:
            print(f"[L.U.N.A. LLM] 요청 중 오류 발생: {e}")
            return ""