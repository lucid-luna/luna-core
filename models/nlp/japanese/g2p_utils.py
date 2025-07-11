# ====================================================================
#  File: models/nlp/japanese/g2p_utils.py
# ====================================================================

from models.nlp.japanese.g2p import g2p
from models.nlp.japanese.mora_list import (
    CONSONANTS,
    MORA_KATA_TO_MORA_PHONEMES,
    MORA_PHONEMES_TO_MORA_KATA
)
from models.nlp.symbols import PUNCTUATIONS

def g2kata_tone(norm_text: str) -> list[tuple[str, int]]:
    """
    텍스트에서 카타카나와 악센트의 쌍 목록을 반환합니다.
    추론 시에만 사용되는 함수이므로 항상 `raise_yomi_error=False`를 지정하여 g2p()를 호출하는 방식입니다.

    Args:
        norm_text: 정규화된 텍스트

    Returns:
        카타카나와 음높이의 리스트
    """

    phones, tones, _ = g2p(norm_text, use_jp_extra=True, raise_yomi_error=False)
    return phone_tone2kata_tone(list(zip(phones, tones)))


def phone_tone2kata_tone(phone_tone: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """
    phone_tone의 phone 부분을 카타카나로 변환합니다. 단, 처음과 마지막의 ("_", 0)은 무시합니다.

    Args:
        phone_tone: 음소와 음높이의 리스트

    Returns:
        카타카나와 음높이의 리스트
    """

    phone_tone = phone_tone[1:]
    phones = [phone for phone, _ in phone_tone]
    tones = [tone for _, tone in phone_tone]
    result: list[tuple[str, int]] = []
    current_mora = ""
    for phone, next_phone, tone, next_tone in zip(phones, phones[1:], tones, tones[1:]):
        if phone in PUNCTUATIONS:
            result.append((phone, tone))
            continue
        if phone in CONSONANTS:
            assert current_mora == "", f"Unexpected {phone} after {current_mora}"
            assert tone == next_tone, f"Unexpected {phone} tone {tone} != {next_tone}"
            current_mora = phone
        else:
            current_mora += phone
            result.append((MORA_PHONEMES_TO_MORA_KATA[current_mora], tone))
            current_mora = ""

    return result


def kata_tone2phone_tone(kata_tone: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """
    `phone_tone2kata_tone()`의 역변환을 수행합니다.

    Args:
        kata_tone: 카타카나와 음높이의 리스트

    Returns:
        음소와 음높이의 리스트
    """

    result: list[tuple[str, int]] = [("_", 0)]
    for mora, tone in kata_tone:
        if mora in PUNCTUATIONS:
            result.append((mora, tone))
        else:
            consonant, vowel = MORA_KATA_TO_MORA_PHONEMES[mora]
            if consonant is None:
                result.append((vowel, tone))
            else:
                result.append((consonant, tone))
                result.append((vowel, tone))
    result.append(("_", 0))

    return result
