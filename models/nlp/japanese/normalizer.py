# ====================================================================
#  File: models/nlp/japanese/normalizer.py
# ====================================================================
"""
일본어 텍스트 정규화 모듈
"""
import re
import unicodedata

from num2words import num2words
from ..symbols import PUNCTUATIONS

# 기호 정규화
__REPLACE_MAP = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "．": ".",
    "…": "...",
    "···": "...",
    "・・・": "...",
    "·": ",",
    "・": ",",
    "、": ",",
    "$": ".",
    "“": "'",
    "”": "'",
    '"': "'",
    "‘": "'",
    "’": "'",
    "（": "'",
    "）": "'",
    "(": "'",
    ")": "'",
    "《": "'",
    "》": "'",
    "【": "'",
    "】": "'",
    "[": "'",
    "]": "'",
    "\u02d7": "\u002d",  # ˗, Modifier Letter Minus Sign
    "\u2010": "\u002d",  # ‐, Hyphen,
    "\u2012": "\u002d",  # ‒, Figure Dash
    "\u2013": "\u002d",  # –, En Dash
    "\u2014": "\u002d",  # —, Em Dash
    "\u2015": "\u002d",  # ―, Horizontal Bar
    "\u2043": "\u002d",  # ⁃, Hyphen Bullet
    "\u2212": "\u002d",  # −, Minus Sign
    "\u23af": "\u002d",  # ⎯, Horizontal Line Extension
    "\u23e4": "\u002d",  # ⏤, Straightness
    "\u2500": "\u002d",  # ─, Box Drawings Light Horizontal
    "\u2501": "\u002d",  # ━, Box Drawings Heavy Horizontal
    "\u2e3a": "\u002d",  # ⸺, Two-Em Dash
    "\u2e3b": "\u002d",  # ⸻, Three-Em Dash
    "「": "'",
    "」": "'",
}
__REPLACE_PATTERN = re.compile('|'.join(re.escape(p) for p in __REPLACE_MAP))

# 허용 문자 이외 정규화
__PUNCTUATION_CLEANUP_PATTERN = re.compile(
    r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF\u3005"
    + r"\u0041-\u005A\u0061-\u007A"
    + r"\uFF21-\uFF3A\uFF41-\uFF5A"
    + r"\u0370-\u03FF\u1F00-\u1FFF"
    + "".join(PUNCTUATIONS) + r"]+",
)

# 숫자 및 통화 기호 정규화
__CURRENCY_MAP = {"$": "ドル", "¥": "円", "£": "ポンド", "€": "ユーロ"}
__CURRENCY_PATTERN = re.compile(r"([$¥£€])([0-9.]*[0-9])")
__NUMBER_PATTERN = re.compile(r"[0-9]+(\.[0-9]+)?")
__NUMBER_WITH_SEPARATOR_PATTERN = re.compile("[0-9]{1,3}(,[0-9]{3})+")

def normalize_text(text: str) -> str:
    """
    일본어 텍스트를 정규화하여 G2P 처리 가능한 형태로 변환합니다.
    숫자를 한자 표기로 변환하고, 구두점 및 불필요 문자를 정리합니다.

    Args:
        text: 원본 텍스트
    Returns:
        str: 정규화된 텍스트
    """
    
    res = unicodedata.normalize('NFKC', text)
    res = __convert_numbers_to_words(res)
    res = res.replace("~", "ー")
    res = res.replace("～", "ー")
    res = res.replace("〜", "ー")
    res = res.replace("\u3099", "")
    res = res.replace("\u309A", "")
    return res

def replace_punctuation(text: str) -> str:
    """
    구두점 및 특수 기호를 '.', ',', '!', '?', "'", '-'로 정규화하고
    허용 문자 이외는 모두 제거합니다.

    Args:
        text: 입력 텍스트
    Returns:
        str: 정규화된 텍스트
    """
    replaced_text = __REPLACE_PATTERN.sub(lambda x: __REPLACE_MAP[x.group()], text)
    replaced_text = __PUNCTUATION_CLEANUP_PATTERN.sub("", replaced_text)
    return replaced_text

def __convert_numbers_to_words(text: str) -> str:
    """
    숫자 및 통화 기호를 일본어 단어로 변환합니다.

    Args:
        text: 입력 텍스트
    Returns:
        str: 변환된 텍스트
    """
    res = __NUMBER_WITH_SEPARATOR_PATTERN.sub(lambda m: m[0].replace(",", ""), text)
    res = __CURRENCY_PATTERN.sub(lambda m: m[2] + __CURRENCY_MAP.get(m[1], m[1]), res)
    res = __NUMBER_PATTERN.sub(lambda m: num2words(m[0], lang="ja"), res)

    return res