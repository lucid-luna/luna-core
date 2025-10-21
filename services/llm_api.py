# ====================================================================
#  File: services/llm_api.py
# ====================================================================

import os
import hashlib
from typing import Dict, List, Any, Optional
from google import genai
from dotenv import load_dotenv
from .cache import ResponseCache

# .env 파일 로드
load_dotenv()

class LLMAPIService:
    def __init__(
        self,
        api_configs: Dict[str, Dict[str, str]],
        enable_cache: bool = True,
        cache_ttl: int = 3600
    ):
        """
        Luna LLM API 서비스 초기화 (Gemini API 지원)

        Args:
            api_configs (Dict[str, Dict[str, str]]): API 제공자별 설정
                예: {
                    "gemini": {
                        "api_key": "your-api-key",
                        "model": "gemini-2.5-flash"
                    }
                }
            enable_cache (bool): 캐싱 활성화 여부
            cache_ttl (int): 캐시 유효 시간 (초)
        """
        self.api_configs = api_configs
        self.clients = {}
        self.enable_cache = enable_cache
        
        # 캐시 초기화
        if enable_cache:
            self.cache = ResponseCache(
                cache_dir="./cache",
                ttl=cache_ttl,
                max_cache_size=100,
                similarity_threshold=0.85
            )
            print("[L.U.N.A. LLM API] 응답 캐싱 활성화됨")
        else:
            self.cache = None
        
        self._initialize_clients()
    
    def _initialize_clients(self):
        """API 클라이언트 초기화"""
        for provider, config in self.api_configs.items():
            if provider == "gemini":
                api_key = config.get("api_key") or os.getenv("GEMINI_API_KEY") or os.getenv("GENAI_API_KEY")
                if api_key:
                    try:
                        self.clients[provider] = genai.Client(api_key=api_key)
                        print(f"[L.U.N.A. LLM API] Gemini 클라이언트 초기화 완료")
                    except Exception as e:
                        print(f"[L.U.N.A. LLM API] Gemini 클라이언트 초기화 실패: {e}")
                else:
                    print(f"[L.U.N.A. LLM API] Gemini API 키가 설정되지 않았습니다.")
    
    def generate(
        self,
        target_provider: str,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        LLM API에 프롬프트를 보내고 응답을 받습니다.

        Args:
            target_provider (str): 사용할 API 제공자 (예: "gemini")
            system_prompt (str, optional): 시스템 프롬프트
            user_prompt (str, optional): 사용자 프롬프트
            messages (List[Dict[str, Any]], optional): 대화 내역
            model (str, optional): 사용할 모델 (기본값은 설정에서 가져옴)
            stream (bool): 스트리밍 모드 활성화

        Returns:
            Dict[str, Any]: API 응답 또는 스트림 제너레이터
        """
        if target_provider not in self.api_configs:
            print(f"[L.U.N.A. LLM API] 알 수 없는 API 제공자입니다: {target_provider}")
            return {"error": "Unknown provider"}

        if target_provider not in self.clients:
            print(f"[L.U.N.A. LLM API] '{target_provider}' 클라이언트가 초기화되지 않았습니다.")
            return {"error": "Client not initialized"}

        config = self.api_configs[target_provider]
        
        if target_provider == "gemini":
            return self._generate_gemini(
                config=config,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                messages=messages,
                model=model,
                stream=stream
            )
        
        return {"error": "Unsupported provider"}
    
    def _generate_gemini(
        self,
        config: Dict[str, str],
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Gemini API를 통해 텍스트를 생성합니다.

        Args:
            config (Dict[str, str]): Gemini 설정
            system_prompt (str, optional): 시스템 프롬프트
            user_prompt (str, optional): 사용자 프롬프트
            messages (List[Dict[str, Any]], optional): 대화 내역
            model (str, optional): 모델 이름
            stream (bool): 스트리밍 모드 활성화

        Returns:
            Dict[str, Any]: Gemini API 응답
        """
        client = self.clients["gemini"]
        model_name = model or config.get("model", "gemini-2.5-flash")
        
        # 프롬프트 구성
        if messages:
            # messages가 있는 경우, 대화 내역을 결합
            parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    parts.append(content)
                elif role == "user":
                    parts.append(f"User: {content}")
                elif role == "assistant":
                    parts.append(f"Assistant: {content}")
            final_prompt = "\n\n".join(parts)
        else:
            # 기본적인 시스템/사용자 프롬프트 구성
            if system_prompt is None or user_prompt is None:
                print("[L.U.N.A. LLM API] 시스템 프롬프트와 사용자 프롬프트는 필수입니다.")
                return {"error": "Missing prompts"}
            
            parts = [system_prompt.strip(), user_prompt.strip()]
            final_prompt = "\n\n".join(parts)
        
        # 캐시 확인 (사용자 입력만 추출)
        user_input = None
        if messages:
            # 마지막 사용자 메시지 추출
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    user_input = msg.get("content", "")
                    break
        elif user_prompt:
            user_input = user_prompt
        
        # 컨텍스트 해시 생성 (대화 히스토리)
        context_hash = ""
        if messages and len(messages) > 1:
            context_str = str(messages[:-1])  # 마지막 메시지 제외
            context_hash = hashlib.md5(context_str.encode('utf-8')).hexdigest()[:8]
        
        # 캐시에서 조회
        if self.cache and user_input:
            cached_response = self.cache.get(
                prompt=user_input,
                model=model_name,
                context_hash=context_hash,
                use_similarity=True
            )
            if cached_response:
                return cached_response
        
        try:
            # 스트리밍 모드
            if stream:
                return self._generate_gemini_stream(
                    client=client,
                    model_name=model_name,
                    final_prompt=final_prompt,
                    user_input=user_input,
                    context_hash=context_hash
                )
            
            # 일반 모드 (기존 로직)
            # Gemini API 호출 (제공하신 코드 방식 그대로)
            response = client.models.generate_content(
                model=model_name,
                contents=final_prompt
            )
            
            # 응답 텍스트 추출 (제공하신 코드 방식)
            text = getattr(response, "text", None) or getattr(response, "content", None) or str(response)
            
            # OpenAI API와 유사한 형식으로 응답 구성
            result = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": text
                        },
                        "finish_reason": "stop"
                    }
                ],
                "model": model_name,
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }
            
            # 캐시에 저장
            if self.cache and user_input:
                self.cache.set(
                    prompt=user_input,
                    response=result,
                    model=model_name,
                    context_hash=context_hash
                )
            
            return result
        
        except Exception as e:
            print(f"[L.U.N.A. LLM API] Gemini API 호출 중 오류 발생: {e}")
            
            # 폴백: 시스템 프롬프트 없이 재시도 (제공하신 코드 방식)
            if system_prompt and user_prompt:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=user_prompt
                    )
                    text = getattr(response, "text", None) or getattr(response, "content", None) or str(response)
                    
                    return {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": text
                                },
                                "finish_reason": "stop"
                            }
                        ],
                        "model": model_name,
                        "usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0
                        }
                    }
                except Exception as fallback_error:
                    print(f"[L.U.N.A. LLM API] 폴백 호출도 실패: {fallback_error}")
                    return {"error": str(fallback_error)}
            
            return {"error": str(e)}
    
    def _generate_gemini_stream(
        self,
        client,
        model_name: str,
        final_prompt: str,
        user_input: Optional[str] = None,
        context_hash: str = ""
    ):
        """
        Gemini API 스트리밍 응답 제너레이터

        Args:
            client: Gemini 클라이언트
            model_name: 모델 이름
            final_prompt: 완성된 프롬프트
            user_input: 사용자 입력 (캐싱용)
            context_hash: 컨텍스트 해시 (캐싱용)

        Yields:
            Dict[str, Any]: 스트리밍 청크 데이터
        """
        try:
            # Gemini API 스트리밍 호출
            response_stream = client.models.generate_content_stream(
                model=model_name,
                contents=final_prompt
            )
            
            full_text = ""
            
            for chunk in response_stream:
                # 청크에서 텍스트 추출
                chunk_text = getattr(chunk, "text", None) or ""
                
                if chunk_text:
                    full_text += chunk_text
                    
                    # OpenAI 스트림 형식으로 변환
                    yield {
                        "choices": [
                            {
                                "delta": {
                                    "role": "assistant",
                                    "content": chunk_text
                                },
                                "finish_reason": None
                            }
                        ],
                        "model": model_name
                    }
            
            # 마지막 청크 (완료 신호)
            yield {
                "choices": [
                    {
                        "delta": {},
                        "finish_reason": "stop"
                    }
                ],
                "model": model_name
            }
            
            # 캐시에 전체 응답 저장
            if self.cache and user_input and full_text:
                complete_response = {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": full_text
                            },
                            "finish_reason": "stop"
                        }
                    ],
                    "model": model_name,
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                }
                self.cache.set(
                    prompt=user_input,
                    response=complete_response,
                    model=model_name,
                    context_hash=context_hash
                )
        
        except Exception as e:
            print(f"[L.U.N.A. LLM API] Gemini 스트리밍 중 오류 발생: {e}")
            yield {"error": str(e)}
    
    def get_available_providers(self) -> List[str]:
        """
        사용 가능한 API 제공자 목록을 반환합니다.

        Returns:
            List[str]: 초기화된 API 제공자 목록
        """
        return list(self.clients.keys())
