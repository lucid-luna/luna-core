# ====================================================================
#  File: models/nlp/bert_models.py
# ====================================================================
"""
일본어 BERT 모델/토크나이저 로드 모듈입니다.

- 한 번 로드된 모델과 토크나이저는 캐시에 저장되어, 이후 호출 시 즉시 반환됩니다.
- DEFAULT_BERT_TOKENIZER_PATHS에 정의된 기본 경로에서 모델을 가져오도록 설계되었습니다.
"""

import gc
import logging
from typing import Optional, Union, cast

import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DebertaV2Model,
    DebertaV2Tokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from models.nlp.constants import DEFAULT_BERT_TOKENIZER_PATHS, Languages

logger = logging.getLogger(__name__)

# 로드된 모델 및 토크나이저 캐시
__loaded_models: dict[Languages, Union[PreTrainedModel, DebertaV2Model]] = {}

__loaded_tokenizers: dict[
    Languages, Union[PreTrainedTokenizer, PreTrainedTokenizerFast, DebertaV2Tokenizer]
] = {}

def load_model(
    language: Languages,
    pretrained_model_name_or_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    revision: str = "main",
) -> Union[PreTrainedModel, DebertaV2Model]:
    """
    지정된 언어의 BERT 모델을 로드합니다. 이미 로드된 경우 캐시를 반환합니다.

    Args:
        language: 대상 언어 (Languages.JP)
        pretrained_model_name_or_path: HF 리포 또는 로컬 경로. None이면 DEFAULT 경로 사용.
        cache_dir: 캐시 디렉토리 (HF 모델일 경우에만 유효).
        revision: HF 리포의 브랜치/태그.

    Returns:
        로드된 PreTrainedModel 또는 DebertaV2Model 인스턴스.
    """
    if language in __loaded_models:
        return __loaded_models[language]
    
    if pretrained_model_name_or_path is None:
        default_path = DEFAULT_BERT_TOKENIZER_PATHS.get(language)
        assert default_path and default_path.exists(), (
            f"[L.U.N.A.] BERT 모델 경로가 정의되지 않았습니다: {language}"
        )
        pretrained_model_name_or_path = str(default_path)

    model: Union[PreTrainedModel, DebertaV2Model]

    model = AutoModelForMaskedLM.from_pretrained(
        pretrained_model_name_or_path,
        cache_dir=cache_dir,
        revision=revision,
    )
    __loaded_models[language] = model
    logger.info(f"[L.U.N.A. NLP] Loaded {language} BERT model from {pretrained_model_name_or_path}")
    return model

def load_tokenizer(
    language: Languages,
    pretrained_model_name_or_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    revision: str = "main",
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast, DebertaV2Tokenizer]:
    """
    지정된 언어의 BERT 토크나이저를 로드합니다. 이미 로드된 경우 캐시를 반환합니다.

    Args:
        language: 대상 언어 (Languages.JP)
        pretrained_model_name_or_path: HF 리포 또는 로컬 경로. None이면 DEFAULT 경로 사용.
        cache_dir: 캐시 디렉토리 (HF 모델일 경우에만 유효).
        revision: HF 리포의 브랜치/태그.

    Returns:
        로드된 토크나이저 인스턴스.
    """
    if language in __loaded_tokenizers:
        return __loaded_tokenizers[language]
    
    if pretrained_model_name_or_path is None:
        default_path = DEFAULT_BERT_TOKENIZER_PATHS.get(language)
        assert default_path and default_path.exists(), (
            f"[L.U.N.A.] BERT 토크나이저 경로가 정의되지 않았습니다: {language}"
        )
        pretrained_model_name_or_path = str(default_path)

    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, DebertaV2Tokenizer]

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        cache_dir=cache_dir,
        revision=revision,
    )
    __loaded_tokenizers[language] = tokenizer
    logger.info(f"[L.U.N.A. NLP] Loaded {language} BERT tokenizer from {pretrained_model_name_or_path}")
    return tokenizer

def unload_model(language: Languages) -> None:
    """
    지정된 언어의 BERT 모델을 언로드합니다.
    """
    if language in __loaded_models:
        del __loaded_models[language]
        gc.collect()
        logger.info(f"[L.U.N.A. NLP] Unloaded {language} BERT model")
        
def unload_tokenizer(language: Languages) -> None:
    """
    지정된 언어의 BERT 토크나이저를 언로드합니다.
    """
    if language in __loaded_tokenizers:
        del __loaded_tokenizers[language]
        gc.collect()
        logger.info(f"[L.U.N.A. NLP] Unloaded {language} BERT tokenizer")
        
def unload_all_models() -> None:
    """
    모든 로드된 BERT 모델을 언로드합니다.
    """
    for language in list(__loaded_models.keys()):
        unload_model(language)
    logger.info("[L.U.N.A. NLP] Unloaded all BERT models")

def unload_all_tokenizers() -> None:
    """
    모든 로드된 BERT 토크나이저를 언로드합니다.
    """
    for language in list(__loaded_tokenizers.keys()):
        unload_tokenizer(language)
    logger.info("[L.U.N.A. NLP] Unloaded all BERT tokenizers")