# ====================================================================
#  File: models/nlp/constants.py
# ====================================================================
"""
일본어 NLP를 위한 상수 정의 파일입니다.
"""

from pathlib import Path
from models.utils.strenum import StrEnum

BASE_DIR = Path(__file__).resolve().parents[2]

class Languages(StrEnum):
    JP = "JP"
    EN = "EN"
    ZH = "ZH"

DEFAULT_BERT_TOKENIZER_PATHS = {
    Languages.JP: BASE_DIR / "bert" / "deberta-v2-large-japanese-char-wwm",
    Languages.EN: BASE_DIR / "bert" / "deberta-v3-large",
    Languages.ZH: BASE_DIR / "bert" / "chinese-roberta-wwm-ext-large",
}

DEFAULT_USER_DICT_DIR = BASE_DIR / "models" / "nlp" / "japanese" / "user_dict"

DEFAULT_STYLE = "Neutral"
DEFAULT_STYLE_WEIGHT = 1.0
DEFAULT_SDP_RATIO = 0.2
DEFAULT_NOISE = 0.6
DEFAULT_NOISEW = 0.8
DEFAULT_LENGTH = 1.0
DEFAULT_LINE_SPLIT = True
DEFAULT_SPLIT_INTERVAL = 0.5
DEFAULT_ASSIST_TEXT_WEIGHT = 0.7
DEFAULT_ASSIST_TEXT_WEIGHT = 1.0
