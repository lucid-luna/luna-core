# ====================================================================
#  File: services/translator.py
# ====================================================================

import os
import time
import uuid
import requests
from typing import Optional
from dotenv import load_dotenv

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
    def __init__(self, session: Optional[requests.Session] = None, logger=None):
        if not AZURE_TRANSLATOR_KEY or not AZURE_TRANSLATOR_ENDPOINT or not AZURE_TRANSLATOR_REGION:
            raise ValueError("[L.U.N.A.] Azure Translator 환경변수가 누락되었습니다.")
        self.session = session or requests.Session()
        self.logger = logger

    def translate(self, text: str, from_lang: str, to_lang: str) -> str:
        try:
            if not text or from_lang == to_lang:
                return text

            url = f"{AZURE_TRANSLATOR_ENDPOINT}/translate"
            params = {
                "api-version": API_VERSION,
                "from": from_lang,
                "to": to_lang,
            }
            headers = {
                "Ocp-Apim-Subscription-Key": AZURE_TRANSLATOR_KEY,
                "Ocp-Apim-Subscription-Region": AZURE_TRANSLATOR_REGION,
                "Content-Type": "application/json",
                "X-ClientTraceId": str(uuid.uuid4()),
            }
            body = [{"text": text}]

            last_err = None
            for attempt in range(RETRY_COUNT + 1):
                try:
                    resp = self.session.post(
                        url, params=params, headers=headers, json=body, timeout=DEFAULT_TIMEOUT
                    )
                    if resp.status_code in (429, 500, 502, 503, 504):
                        raise requests.HTTPError(f"Retryable status: {resp.status_code}", response=resp)
                    resp.raise_for_status()
                    data = resp.json()
                    return data[0]["translations"][0]["text"]
                except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
                    last_err = e
                    retryable = isinstance(e, (requests.Timeout, requests.ConnectionError)) \
                                or (isinstance(e, requests.HTTPError) and resp is not None and resp.status_code in (429, 500, 502, 503, 504))
                    if attempt < RETRY_COUNT and retryable:
                        if self.logger:
                            self.logger.warning(f"[Translator] 시도 {attempt+1} 실패({type(e).__name__}), 재시도 준비")
                        time.sleep(RETRY_BACKOFF * (attempt + 1))
                        continue
                    break

            if self.logger:
                self.logger.warning(f"[Translator] 최종 실패, 원문 반환: {last_err}")
            return text

        except Exception as e:
            if self.logger:
                self.logger.warning(f"[Translator] 예외 발생, 원문 반환: {e}")
            return text