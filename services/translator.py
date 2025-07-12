# ====================================================================
#  File: services/translator.py
# ====================================================================

import os
import requests
import uuid
from dotenv import load_dotenv

load_dotenv()

# 키 및 엔드포인트 설정
AZURE_TRANSLATOR_KEY = os.getenv("AZURE_TRANSLATOR_KEY")
AZURE_TRANSLATOR_ENDPOINT = os.getenv("AZURE_TRANSLATOR_ENDPOINT")
AZURE_TRANSLATOR_REGION = os.getenv("AZURE_TRANSLATOR_REGION")
API_VERSION = "3.0"

class TranslatorService:
    def __init__(self):
        if not AZURE_TRANSLATOR_KEY or not AZURE_TRANSLATOR_ENDPOINT or not AZURE_TRANSLATOR_REGION:
            raise ValueError("[L.U.N.A.] Azure Translator 환경변수가 누락되었습니다.")

    def translate(self, text: str, from_lang: str, to_lang: str) -> str:
        url = f"{AZURE_TRANSLATOR_ENDPOINT}/translate?api-version={API_VERSION}&from={from_lang}&to={to_lang}"

        headers = {
            "Ocp-Apim-Subscription-Key": AZURE_TRANSLATOR_KEY,
            "Ocp-Apim-Subscription-Region": AZURE_TRANSLATOR_REGION,
            "Content-Type": "application/json",
            "X-ClientTraceId": str(uuid.uuid4())
        }

        body = [{"text": text}]
        response = requests.post(url, headers=headers, json=body)
        response.raise_for_status()
        result = response.json()

        return result[0]["translations"][0]["text"]