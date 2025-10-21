# ====================================================================
#  File: services/vision.py
# ====================================================================
"""
LunaVision 서비스 모듈

- config/models.yaml 의 vision 항목을 읽어와
    VLMPipeline 을 초기화합니다.
- predict(image_bytes: bytes) 호출 시,
    템플릿 기반 prompt+이미지로 추론하고 결과 문자열을 반환합니다.
"""

import io
from pathlib import Path
import asyncio
import numpy as np
from PIL import Image
from utils.config import load_config_dict
from openvino import Tensor
from openvino_genai import VLMPipeline, GenerationConfig

def pil_to_tensor(img: Image.Image) -> Tensor:
    # HWC -> 1x HWC, uint8
    return Tensor(np.asarray(img, dtype=np.uint8)[None])

class VisionService:
    def __init__(self):
        config = load_config_dict("models")["vision"]
        
        self.model_dir = Path(config["model_dir"]).expanduser()
        self.device = config.get("device", "AUTO")
        self.max_new_tokens = config.get("max_new_tokens", 16)
        self.char_limit = config.get("char_limit", 256)
        self._prompt_template = config.get("prompt_tpl", "Describe the image: {}")
        
        self._semaphore = asyncio.Semaphore(config.get("concurrency", 1))

        self._pipe = VLMPipeline(str(self.model_dir), device=self.device)

        tokenizer = self._pipe.get_tokenizer()
        self.eos_id = getattr(tokenizer, "eos_token_id", None) or tokenizer.get_eos_token_id()
        
    async def predict(self, image_bytes: bytes) -> str:
        """
        bytes -> PIL.Image
            -> prompt 생성
            -> VLMPipeline.generate 호출 (max_new_tokens, eos_id 세팅)
            -> 후처리 (user 토큰 제거, 길이 제한)
        """
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        gconfig = GenerationConfig(
            eos_token_id=self.eos_id,
            pad_token_id=self.eos_id,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=0.0
        )
        
        async with self._semaphore:
            tensor = pil_to_tensor(image)
            
            result = await asyncio.to_thread(
                lambda: self._pipe.generate(
                    self._prompt_template,
                    images=[tensor],
                    generation_config=gconfig
                )
            )
            
        answer = result.text if hasattr(result, "text") else str(result)
        if "<|user|>" in answer:
            answer = answer.split("<|user|>")[0].strip()
        if len(answer) > self.char_limit:
            answer = answer[: self.char_limit].rsplit(" ", 1)[0] + " ..."
        return answer