# ====================================================================
#  File: models/nlp/__init__.py
# ====================================================================
"""
일본어 음소(symbol) 및 어조(tone) 관련 상수를 정의합니다.
"""

from .constants import Languages

PAD = "_"

PUNCTUATIONS = ["!", "?", "…", ",", ".", "'", "-"]
PUNCTUATION_SYMBOLS = PUNCTUATIONS + ["SP", "UNK"]

JP_SYMBOLS = [
    "N", "a", "a:", "b", "by", "ch", "d", "dy", "e", "e:",
    "f", "g", "gy", "h", "hy", "i", "i:", "j", "k", "ky",
    "m", "my", "n", "ny", "o", "o:", "p", "py", "q", "r",
    "ry", "s", "sh", "t", "ts", "ty", "u", "u:", "w", "y",
    "z", "zy",
]

SYMBOLS = [PAD] + JP_SYMBOLS + PUNCTUATION_SYMBOLS

LANGUAGE_TONE_START_MAP = {
    Languages.JP: 0,
}

LANGUAGE_ID_MAP = {
    Languages.JP: 0,
}
