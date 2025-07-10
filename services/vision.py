# ====================================================================
#  File: services/vision.py
# ====================================================================
"""
LunaVision 서비스 모듈

- config/models.yaml 의 vision 항목을 읽어와
    VLMPipeline 또는 OVMS(gRPC) 클라이언트를 초기화합니다.
- analyze(file: UploadFile) 호출 시 이미지 내 OCR + 추론을 수행합니다.
"""

from pathlib import Path
from io import BytesIO
from typing import Optional

import numpy as np
from PIL import Image
from fastapi import UploadFile, HTTPException
from openvino import Tensor
from openvino_genai import VLMPipeline, GenerationConfig
from pytesseract import image_to_string

from utils.config import load_config_dict