# ====================================================================
#  File: services/translator.py
# ====================================================================

import os
import time
import uuid
import requests
from typing import Optional
from dotenv import load_dotenv
from functools import lru_cache

load_dotenv()

# 키 및 엔드포인트 설정
AZURE_TRANSLATOR_KEY = os.getenv("AZURE_TRANSLATOR_KEY")
AZURE_TRANSLATOR_ENDPOINT = os.getenv("AZURE_TRANSLATOR_ENDPOINT")
AZURE_TRANSLATOR_REGION = os.getenv("AZURE_TRANSLATOR_REGION")
API_VERSION = "3.0"

DEFAULT_TIMEOUT = 6.0
RETRY_COUNT = 2
RETRY_BACKOFF = 0.6

class TranslatorService:
    """
    Azure Translator API 래퍼
    
    - 재시도 로직
    - 비속어 필터링 비활성화
    - 로깅 지원
    """
    
    def __init__(self, session: Optional[requests.Session] = None, logger=None):
        if not AZURE_TRANSLATOR_KEY or not AZURE_TRANSLATOR_ENDPOINT or not AZURE_TRANSLATOR_REGION:
            raise ValueError("[Translator] Azure Translator 환경변수가 누락되었습니다.")
        
        self.session = session or requests.Session()
        self.logger = logger
        
        # 비속어 필터링 비활성화
        self.profanity_action = "NoAction"

    def translate(
        self, 
        text: str, 
        from_lang: str, 
        to_lang: str,
        profanity_action: Optional[str] = None
    ) -> str:
        """
        텍스트를 번역합니다.
        
        Args:
            text (str): 번역할 텍스트
            from_lang (str): 원본 언어 코드 (예: 'ko', 'en')
            to_lang (str): 목표 언어 코드
            profanity_action (str, optional): 비속어 처리 방식 (기본: NoAction)
            
        Returns:
            str: 번역된 텍스트 (실패 시 원문 반환)
        """
        try:
            # 입력 검증
            if not text or not text.strip():
                return text
            
            # 같은 언어면 번역 생략
            if from_lang == to_lang:
                return text

            url = f"{AZURE_TRANSLATOR_ENDPOINT}/translate"
            params = {
                "api-version": API_VERSION,
                "from": from_lang,
                "to": to_lang,
                "profanityAction": profanity_action or self.profanity_action,
            }
            headers = {
                "Ocp-Apim-Subscription-Key": AZURE_TRANSLATOR_KEY,
                "Ocp-Apim-Subscription-Region": AZURE_TRANSLATOR_REGION,
                "Content-Type": "application/json",
                "X-ClientTraceId": str(uuid.uuid4()),
            }
            body = [{"text": text}]

            last_err = None
            resp = None
            
            # 재시도 로직
            for attempt in range(RETRY_COUNT + 1):
                try:
                    resp = self.session.post(
                        url, 
                        params=params, 
                        headers=headers, 
                        json=body, 
                        timeout=DEFAULT_TIMEOUT
                    )
                    
                    # 재시도 가능한 상태 코드
                    if resp.status_code in (429, 500, 502, 503, 504):
                        raise requests.HTTPError(
                            f"Retryable status: {resp.status_code}", 
                            response=resp
                        )
                    
                    resp.raise_for_status()
                    data = resp.json()
                    
                    translated = data[0]["translations"][0]["text"]
                    
                    if self.logger:
                        self.logger.info(
                            f"[Translator] '{text[:30]}...' ({from_lang}) -> "
                            f"'{translated[:30]}...' ({to_lang})"
                        )
                    
                    return translated
                    
                except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
                    last_err = e
                    retryable = isinstance(e, (requests.Timeout, requests.ConnectionError)) \
                                or (isinstance(e, requests.HTTPError) and resp is not None 
                                    and resp.status_code in (429, 500, 502, 503, 504))
                    
                    if attempt < RETRY_COUNT and retryable:
                        if self.logger:
                            self.logger.warning(
                                f"[Translator] 시도 {attempt+1} 실패({type(e).__name__}), "
                                f"재시도 준비 (대기: {RETRY_BACKOFF * (attempt + 1)}초)"
                            )
                        time.sleep(RETRY_BACKOFF * (attempt + 1))
                        continue
                    break

            # 모든 재시도 실패
            if self.logger:
                self.logger.warning(
                    f"[Translator] 최종 실패, 원문 반환: {last_err}"
                )
            return text

        except Exception as e:
            if self.logger:
                self.logger.warning(
                    f"[Translator] 예외 발생, 원문 반환: {e}"
                )
            return text
    
    def translate_batch(
        self,
        texts: list[str],
        from_lang: str,
        to_lang: str
    ) -> list[str]:
        """
        여러 텍스트를 한 번에 번역합니다.
        
        Args:
            texts (list[str]): 번역할 텍스트 리스트
            from_lang (str): 원본 언어 코드
            to_lang (str): 목표 언어 코드
            
        Returns:
            list[str]: 번역된 텍스트 리스트 (실패 시 원문 반환)
        """
        if not texts:
            return []
        
        try:
            url = f"{AZURE_TRANSLATOR_ENDPOINT}/translate"
            params = {
                "api-version": API_VERSION,
                "from": from_lang,
                "to": to_lang,
                "profanityAction": self.profanity_action,
            }
            headers = {
                "Ocp-Apim-Subscription-Key": AZURE_TRANSLATOR_KEY,
                "Ocp-Apim-Subscription-Region": AZURE_TRANSLATOR_REGION,
                "Content-Type": "application/json",
                "X-ClientTraceId": str(uuid.uuid4()),
            }
            body = [{"text": t} for t in texts]
            
            resp = self.session.post(
                url,
                params=params,
                headers=headers,
                json=body,
                timeout=DEFAULT_TIMEOUT * 2  # 배치는 더 긴 타임아웃
            )
            resp.raise_for_status()
            
            data = resp.json()
            return [item["translations"][0]["text"] for item in data]
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"[Translator] 배치 번역 실패: {e}")
            return texts
        
    @lru_cache(maxsize=1000)
    def translate_cached(self, text: str, from_lang: str, to_lang: str) -> str:
        """캐싱된 번역 (동일 입력 재사용)"""
        return self.translate(text, from_lang, to_lang)