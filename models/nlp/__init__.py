# =============================================================
#  LunaTTS – Text-to-Speech Model
#  This single file shows **all** key modules that should live under
#  models/nlp/ according to the directory layout we agreed on.
# =============================================================

# ====================================================================
#  File: models/nlp/__init__.py
# ====================================================================

from typing import TYPE_CHECKING, Optional

from models.nlp.constants import Languages
from models.nlp.symbols import SYMBOLS, LANGUAGE_ID_MAP, LANGUAGE_TONE_START_MAP

if TYPE_CHECKING:
    import torch
    
__symbol_to_id = {s: i for i, s in enumerate(SYMBOLS)}

def extract_bert_feature(
    text: str,
    word2ph: list[int],
    language: Languages,
    device: str,
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
) -> "torch.Tensor":
    """
    일본어 텍스트에서 BERT 특징(feature)을 추출합니다.

    Args:
        text (str): 텍스트
        word2ph (list[int]): 원본 텍스트의 각 문자에 음소가 몇 개 할당되는지를 나타내는 리스트
        language (Languages): 텍스트의 언어
        device (str): 추론에 사용할 장치
        assist_text (Optional[str], optional): 보조 텍스트 (기본값: None)
        assist_text_weight (float, optional): 보조 텍스트의 가중치 (기본값: 0.7)

    Returns:
        Tensor: 추출된 BERT 특징
    """
    from models.nlp.japanese.bert_feature import extract_bert_feature
    
    return extract_bert_feature(
        text=text,
        word2ph=word2ph,
        device=device,
        assist_text=assist_text,
        assist_text_weight=assist_text_weight,
    )

def clean_text(
    text: str,
    language: Languages,
    use_jp_extra: bool = True,
    raise_yomi_error: bool = False,
) -> tuple[str, list[str], list[int], list[int]]:
    """
    텍스트를 클리닝하고, 음소로 변환합니다.

    Args:
        text (str): 클리닝할 텍스트
        language (Languages): 텍스트의 언어
        use_jp_extra (bool, optional): 텍스트가 일본어인 경우 JP-Extra 모델을 사용할지 여부 (기본값: True)
        raise_yomi_error (bool, optional): False인 경우, 읽을 수 없는 문자가 사라진 것처럼 처리됩니다. (기본값: False)

    Returns:
        tuple[str, list[str], list[int], list[int]]: 클리닝된 텍스트와, 음소・악센트・원본 텍스트의 각 문자에 음소가 몇 개 할당되는지를 나타내는 리스트
    """

    from models.nlp.japanese.g2p import g2p
    from models.nlp.japanese.normalizer import normalize_text

    norm_text = normalize_text(text)
    phones, tones, word2ph = g2p(norm_text, use_jp_extra, raise_yomi_error)

    return norm_text, phones, tones, word2ph


def cleaned_text_to_sequence(
    cleaned_phones: list[str], tones: list[int], language: Languages
) -> tuple[list[int], list[int], list[int]]:
    """
    음소 리스트・악센트 리스트・언어를, 텍스트 내의 대응하는 ID로 변환합니다.

    Args:
        cleaned_phones (list[str]): clean_text()에서 클리닝된 음소의 리스트
        tones (list[int]): 각 음소의 악센트
        language (Languages): 텍스트의 언어

    Returns:
        tuple[list[int], list[int], list[int]]: 텍스트의 기호에 해당하는 정수 목록
    """

    phones = [__symbol_to_id[symbol] for symbol in cleaned_phones]
    tone_start = LANGUAGE_TONE_START_MAP[language]
    tones = [i + tone_start for i in tones]
    lang_id = LANGUAGE_ID_MAP[language]
    lang_ids = [lang_id for i in phones]

    return phones, tones, lang_ids
