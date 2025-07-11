# ====================================================================
#  File: models/nlp/constants.py
# ====================================================================
"""
일본어 NLP를 위한 상수 정의 파일입니다.
"""

from pathlib import Path
from enum import Enum

BASE_DIR = Path(__file__).parent.parent

class Languages(str, Enum):
    JP = "JP"

DEFAULT_BERT_TOKENIZER_PATHS = {
    Languages.JP: BASE_DIR / "models" / "nlp" / "bert" / "deberta-v2-large-japanese-char-wwm"
}

DEFAULT_USER_DICT_DIR = BASE_DIR / "models" / "nlp" / "japanese" / "user_dict"

DEFAULT_SDP_RATIO = 0.2
DEFAULT_NOISE = 0.6
DEFAULT_NOISEW = 0.8
DEFAULT_LENGTH = 1.0
DEFAULT_LINE_SPLIT = True
DEFAULT_SPLIT_INTERVAL = 0.5
DEFAULT_ASSIST_TEXT_WEIGHT = 0.7
