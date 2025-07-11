# ====================================================================
#  File: models/nlp/japanese/g2p.py
# ====================================================================
"""
일본어 G2P(문자→음소+어조) 모듈입니다.
OpenJTalk 워커를 통해 발음·어조 정보를 추출하고,
카타카나 분절 및 Mora 리스트를 이용해
음소 리스트, 어조 리스트, word2ph 매핑을 생성합니다.
"""

import re
import logging
from typing import TypedDict

from models.nlp.constants import Languages
from models.nlp import bert_models
from models.nlp.japanese import pyopenjtalk_worker as pyopenjtalk
from models.nlp.japanese.mora_list import MORA_KATA_TO_MORA_PHONEMES, VOWELS
from models.nlp.japanese.normalizer import replace_punctuation
from models.nlp.symbols import PUNCTUATIONS

logger = logging.getLogger(__name__)

def g2p(
    norm_text: str, use_jp_extra: bool = True, raise_yomi_error: bool = False
) -> tuple[list[str], list[int], list[int]]:
    """
    정규화된 일본어 텍스트(norm_text)를 받아서,
    - phones: 음소 리스트 (구두점 포함)
    - tones: 0 또는 1의 어조 리스트
    - word2ph: 각 원문 문자마다 할당된 음소 개수 리스트
    를 반환합니다.
    """
    
    phone_tone_list_wo_punct = __g2phone_tone_wo_punct(norm_text)
    
    sep_text, sep_kata = text_to_sep_kata(norm_text, raise_yomi_error=raise_yomi_error)
    
    sep_phonemes = __handle_long([__kata_to_phoneme_list(i) for i in sep_kata])
    
    phone_w_punct: list[str] = []
    for i in sep_phonemes:
        phone_w_punct += i
        
    phone_tone_list = __align_tones(phone_w_punct, phone_tone_list_wo_punct)

    sep_tokenized: list[list[str]] = []
    for i in sep_text:
        if i not in PUNCTUATIONS:
            sep_tokenized.append(
                bert_models.load_tokenizer(Languages.JP).tokenize(i)
            )
        else:
            sep_tokenized.append([i])

    word2ph = []
    for token, phoneme in zip(sep_tokenized, sep_phonemes):
        phone_len = len(phoneme)
        word_len = len(token)
        word2ph += __distribute_phone(phone_len, word_len)

    phone_tone_list = [("_", 0)] + phone_tone_list + [("_", 0)]
    word2ph = [1] + word2ph + [1]

    phones = [phone for phone, _ in phone_tone_list]
    tones = [tone for _, tone in phone_tone_list]

    assert len(phones) == sum(word2ph), f"{len(phones)} != {sum(word2ph)}"

    if not use_jp_extra:
        phones = [phone if phone != "N" else "n" for phone in phones]

    return phones, tones, word2ph

def text_to_sep_kata(
    norm_text: str, raise_yomi_error: bool = False
) -> tuple[list[str], list[str]]:
    """
    'normalize_text'로 정규화된 'norm_text'를 받아와 단어를 분할한 뒤,
    분할된 단어 목록과 그에 대응하는 읽기(카타카나 또는 기호 1문자)의 리스트의 튜플을 반환합니다.
    단어 분할 결과는, 'g2p()'의 'word2ph'에서 1문자 당 할당할 음소 기호의 수를 결정하는 데 사용됩니다.
    예시:
    `私はそう思う!って感じ?` →
    ["私", "は", "そう", "思う", "!", "って", "感じ", "?"], ["ワタシ", "ワ", "ソー", "オモウ", "!", "ッテ", "カンジ", "?"]

    Args:
        norm_text (str): 정규화된 일본어 텍스트
        raise_yomi_error (bool, optional): False의 경우 읽을 수 없는 문자가「'」로 발음 (기본값: False)

    Returns:
        tuple[list[str], list[str]]: 분할된 단어 목록과 그에 대응하는 읽기(카타카나 또는 기호 1문자)의 리스트
    """

    parsed = pyopenjtalk.run_frontend(norm_text)
    sep_text: list[str] = []
    sep_kata: list[str] = []

    for parts in parsed:
        word, yomi = replace_punctuation(parts["string"]), parts["pron"].replace(
            "’", ""
        )
        assert yomi != "", f"Empty yomi: {word}"
        if yomi == "、":
            if not set(word).issubset(set(PUNCTUATIONS)):
                if raise_yomi_error:
                    raise YomiError(f"[L.U.N.A.] Cannot read: {word} in:\n{norm_text}")
                logger.warning(
                    f'[L.U.N.A.] Cannot read: {word} in:\n{norm_text}, replaced with "\'"'
                )
                yomi = "'" * len(word)
            else:
                yomi = word
        elif yomi == "？":
            assert word == "?", f"yomi `？` comes from: {word}"
            yomi = "?"
        sep_text.append(word)
        sep_kata.append(yomi)

    return sep_text, sep_kata

def adjust_word2ph(
    word2ph: list[int],
    generated_phone: list[str],
    given_phone: list[str],
) -> list[int]:
    """
    'g2p()'로 얻은 'word2ph'를 generated_phone과 given_phone의 차분 정보를 사용하여 조정합니다.
    generated_phone은 정규화된 읽기 텍스트에서 생성된 읽기 정보이며,
    given_phone은 동일한 읽기 텍스트에 대해 다른 읽기가 제공된 경우, 정규화된 읽기 텍스트의 각 문자에
    음소가 몇 글자 할당되는지를 나타내는 word2ph의 총합이 given_phone의 길이(음소 수)와 일치하지 않을 수 있습니다.
    따라서 generated_phone과 given_phone의 차분을 취하여 변경된 부분에 해당하는 word2ph의 요소 값만 증가 또는 감소시켜,
    악센트에 대한 영향을 최소화하면서 word2ph의 총합을 given_phone의 길이(음소 수)에 일치시킵니다.

    Args:
        word2ph (list[int]): 단어별 음소 수의 리스트
        generated_phone (list[str]): 생성된 음소의 리스트
        given_phone (list[str]): 주어진 음소의 리스트

    Returns:
        list[int]: 수정된 word2ph의 리스트
    """

    word2ph = word2ph[1:-1]
    generated_phone = generated_phone[1:-1]
    given_phone = given_phone[1:-1]

    class DiffDetail(TypedDict):
        begin_index: int
        end_index: int
        value: list[str]

    class Diff(TypedDict):
        generated: DiffDetail
        given: DiffDetail

    def extract_differences(
        generated_phone: list[str], given_phone: list[str]
    ) -> list[Diff]:
        """
        가장 긴 공통 부분 수열(LCS)을 사용하여 두 리스트의 차이를 추출합니다.
        """

        def longest_common_subsequence(
            X: list[str], Y: list[str]
        ) -> list[tuple[int, int]]:
            """
            두 리스트의 가장 긴 공통 부분 수열의 인덱스 쌍을 반환합니다.
            """
            m, n = len(X), len(Y)
            L = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if X[i - 1] == Y[j - 1]:
                        L[i][j] = L[i - 1][j - 1] + 1
                    else:
                        L[i][j] = max(L[i - 1][j], L[i][j - 1])
            index_pairs = []
            i, j = m, n
            while i > 0 and j > 0:
                if X[i - 1] == Y[j - 1]:
                    index_pairs.append((i - 1, j - 1))
                    i -= 1
                    j -= 1
                elif L[i - 1][j] >= L[i][j - 1]:
                    i -= 1
                else:
                    j -= 1
            index_pairs.reverse()
            return index_pairs

        differences = []
        common_indices = longest_common_subsequence(generated_phone, given_phone)
        prev_x, prev_y = -1, -1

        for x, y in common_indices:
            diff_X = {
                "begin_index": prev_x + 1,
                "end_index": x,
                "value": generated_phone[prev_x + 1 : x],
            }
            diff_Y = {
                "begin_index": prev_y + 1,
                "end_index": y,
                "value": given_phone[prev_y + 1 : y],
            }
            if diff_X or diff_Y:
                differences.append({"generated": diff_X, "given": diff_Y})
            prev_x, prev_y = x, y
        if prev_x < len(generated_phone) - 1 or prev_y < len(given_phone) - 1:
            differences.append(
                {
                    "generated": {
                        "begin_index": prev_x + 1,
                        "end_index": len(generated_phone) - 1,
                        "value": generated_phone[prev_x + 1 : len(generated_phone) - 1],
                    },
                    "given": {
                        "begin_index": prev_y + 1,
                        "end_index": len(given_phone) - 1,
                        "value": given_phone[prev_y + 1 : len(given_phone) - 1],
                    },
                }
            )
        for diff in differences[:]:
            if (
                len(diff["generated"]["value"]) == 0
                and len(diff["given"]["value"]) == 0
            ):
                differences.remove(diff)

        return differences

    differences = extract_differences(generated_phone, given_phone)

    adjusted_word2ph: list[int] = [0] * len(word2ph)
    # 現在処理中の generated_phone のインデックス
    current_generated_index = 0

    for word2ph_element_index, word2ph_element in enumerate(word2ph):

        for _ in range(word2ph_element):
            current_diff: Diff | None = None
            for diff in differences:
                if diff["generated"]["begin_index"] == current_generated_index:
                    current_diff = diff
                    break
            if current_diff is not None:
                diff_in_phonemes = \
                    len(current_diff["given"]["value"]) - len(current_diff["generated"]["value"])
                adjusted_word2ph[word2ph_element_index] += diff_in_phonemes
            adjusted_word2ph[word2ph_element_index] += 1
            current_generated_index += 1

    assert len(given_phone) == sum(adjusted_word2ph), f"{len(given_phone)} != {sum(adjusted_word2ph)}"

    for adjusted_word2ph_element_index, adjusted_word2ph_element in enumerate(adjusted_word2ph):
        if adjusted_word2ph_element < 1:
            diff = 1 - adjusted_word2ph_element
            adjusted_word2ph[adjusted_word2ph_element_index] = 1
            for i in range(1, len(adjusted_word2ph)):
                if adjusted_word2ph_element_index + i >= len(adjusted_word2ph):
                    break
                if adjusted_word2ph[adjusted_word2ph_element_index + i] - diff >= 1:
                    adjusted_word2ph[adjusted_word2ph_element_index + i] -= diff
                    break
                else:
                    diff -= adjusted_word2ph[adjusted_word2ph_element_index + i] - 1
                    adjusted_word2ph[adjusted_word2ph_element_index + i] = 1
                    if diff == 0:
                        break

    for adjusted_word2ph_element_index, adjusted_word2ph_element in enumerate(adjusted_word2ph):
        if adjusted_word2ph_element > 6:
            diff = adjusted_word2ph_element - 6
            adjusted_word2ph[adjusted_word2ph_element_index] = 6
            for i in range(1, len(adjusted_word2ph)):
                if adjusted_word2ph_element_index + i >= len(adjusted_word2ph):
                    break
                if adjusted_word2ph[adjusted_word2ph_element_index + i] + diff <= 6:
                    adjusted_word2ph[adjusted_word2ph_element_index + i] += diff
                    break
                else:
                    diff -= 6 - adjusted_word2ph[adjusted_word2ph_element_index + i]
                    adjusted_word2ph[adjusted_word2ph_element_index + i] = 6
                    if diff == 0:
                        break

    return [1] + adjusted_word2ph + [1]
    
def __g2phone_tone_wo_punct(text: str) -> list[tuple[str, int]]:
    """
    텍스트에 대해 음소와 어조(0 또는 1) 쌍의 리스트를 반환합니다.
    단, "! “.” “?” 등의 비음소 기호(punctuation)는 제외합니다.
    비음소 기호를 포함하는 처리는 `align_tones()`에서 수행합니다.
    또한 'っ'은 'q'로, 'ん'은 'N'으로 변환됩니다.
    
    예시: "こんにちは、世界ー。。元気？！"
    [('k', 0), ('o', 0), ('N', 1), ('n', 1), ('i', 1), ('ch', 1), ('i', 1), ('w', 1), ('a', 1), ('s', 1), ('e', 1), ('k', 0), ('a', 0), ('i', 0), ('i', 0), ('g', 1), ('e', 1), ('N', 0), ('k', 0), ('i', 0)]
    
    Args:
        text (str): 정규화된 일본어 텍스트
    
    Returns:
        list[tuple[str, int]]: 음소와 어조 쌍의 리스트
    """
    
    prosodies = __pyopenjtalk_g2p_prosody(text, drop_unvoiced_vowels=True)
    result: list[tuple[str, int]] = []
    current_phrase: list[tuple[str, int]] = []
    current_tone = 0
    
    for i, letter in enumerate(prosodies):
        if letter == "^":
            assert i == 0, "Unexpected ^"
        elif letter in ("$", "?", "_", "#"):
            result.extend(__fix_phone_tone(current_phrase))
            if letter in ("$", "?"):
                assert i == len(prosodies) - 1, f"Unexpected {letter}"
            current_phrase = []
            current_tone = 0
        elif letter == "[":
            current_tone = current_tone + 1
        elif letter == "]":
            current_tone = current_tone - 1
        else:
            if letter == "cl":
                letter = "q"
            current_phrase.append((letter, current_tone))
    
    return result

__PYOPENJTALK_G2P_PROSODY_A1_PATTERN = re.compile(r"/A:([0-9\-]+)\+")
__PYOPENJTALK_G2P_PROSODY_A2_PATTERN = re.compile(r"\+(\d+)\+")
__PYOPENJTALK_G2P_PROSODY_A3_PATTERN = re.compile(r"\+(\d+)/")
__PYOPENJTALK_G2P_PROSODY_E3_PATTERN = re.compile(r"!(\d+)_")
__PYOPENJTALK_G2P_PROSODY_F1_PATTERN = re.compile(r"/F:(\d+)_")
__PYOPENJTALK_G2P_PROSODY_P3_PATTERN = re.compile(r"\-(.*?)\+")

def __pyopenjtalk_g2p_prosody(
    text: str, drop_unvoiced_vowels: bool = True
) -> list[str]:
    """
    ESPnet 구현에서 인용,「ん」은 「N」로 처리됨에 유의.
    ref: https://github.com/espnet/espnet/blob/master/espnet2/text/phoneme_tokenizer.py
    ------------------------------------------------------------------------------------------

    입력된 전체 문맥에서 음소 + 운율 기호를 추출합니다.
    
    The algorithm is based on `Prosodic features control by symbols as input of
    sequence-to-sequence acoustic modeling for neural TTS`_ with some r9y9's tweaks.

    Args:
        text (str): 입력 텍스트
        drop_unvoiced_vowels (bool): 무음 모음을 제거할지 여부

    Returns:
        List[str]: 음소와 어조 기호의 리스트

    예시:
        >>> from espnet2.text.phoneme_tokenizer import pyopenjtalk_g2p_prosody
        >>> pyopenjtalk_g2p_prosody("こんにちは。")
        ['^', 'k', 'o', '[', 'N', 'n', 'i', 'ch', 'i', 'w', 'a', '$']

    .. _`Prosodic features control by symbols as input of sequence-to-sequence acoustic
        modeling for neural TTS`: https://doi.org/10.1587/transinf.2020EDP7104
    """
    
    def _numeric_feature_by_regex(pattern: re.Pattern[str], s: str) -> int:
        match = pattern.search(s)
        if match is None:
            return -50
        return int(match.group(1))

    labels = pyopenjtalk.make_label(pyopenjtalk.run_frontend(text))
    N = len(labels)

    phones = []
    for n in range(N):
        lab_curr = labels[n]
        
        p3 = __PYOPENJTALK_G2P_PROSODY_P3_PATTERN.search(lab_curr).group(1)
        
        if drop_unvoiced_vowels and p3 in "AEIOU":
            p3 = p3.lower()
            
        if p3 == "sil":
            assert n == 0 or n == N - 1
            if n == 0:
                phones.append("^")
            elif n == N - 1:
                # check question form or not
                e3 = _numeric_feature_by_regex(
                    __PYOPENJTALK_G2P_PROSODY_E3_PATTERN, lab_curr
                )
                if e3 == 0:
                    phones.append("$")
                elif e3 == 1:
                    phones.append("?")
            continue
        elif p3 == "pau":
            phones.append("_")
            continue
        else:
            phones.append(p3)
        
        a1 = _numeric_feature_by_regex(__PYOPENJTALK_G2P_PROSODY_A1_PATTERN, lab_curr)
        a2 = _numeric_feature_by_regex(__PYOPENJTALK_G2P_PROSODY_A2_PATTERN, lab_curr)
        a3 = _numeric_feature_by_regex(__PYOPENJTALK_G2P_PROSODY_A3_PATTERN, lab_curr)
        
        f1 = _numeric_feature_by_regex(__PYOPENJTALK_G2P_PROSODY_F1_PATTERN, lab_curr)
        
        a2_next = _numeric_feature_by_regex(
            __PYOPENJTALK_G2P_PROSODY_A2_PATTERN, labels[n + 1]
        )

        if a3 == 1 and a2_next == 1 and p3 in "aeiouAEIOUNcl":
            phones.append("#")
        elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
            phones.append("]")
        elif a2 == 1 and a2_next == 2:
            phones.append("[")
            
    return phones

def __fix_phone_tone(phone_tone_list: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """
    'phone_tone_list'의 tone(악센트 값)을 0과 1 사이로 조정합니다.
    예시: [(a, 0), (i, -1), (u, -1)] → [(a, 1), (i, 0), (u, 0)]

    Args:
        phone_tone_list (list[tuple[str, int]]): 음소와 어조 쌍의 리스트

    Returns:
        list[tuple[str, int]]: 수정된 음소와 어조 쌍의 리스트
    """
    
    tone_values = set(tone for _, tone in phone_tone_list)
    if len(tone_values) == 1:
        assert tone_values == {0}, tone_values
        return phone_tone_list
    elif len(tone_values) == 2:
        if tone_values == {0, 1}:
            return phone_tone_list
        elif tone_values == {-1, 0}:
            return [
                (letter, 0 if tone == -1 else 1) for letter, tone in phone_tone_list
            ]
        else:
            raise ValueError(f"[L.U.N.A.] 예상치 못한 악센트 값: {tone_values}")
    else:
        raise ValueError(f"[L.U.N.A.] 예상치 못한 악센트 값: {tone_values}")

def __handle_long(sep_phonemes: list[list[str]]) -> list[list[str]]:
    """ 
    구문별로 구분된 음소(장음기호 그대로) 목록 `sep_phonemes`를 받아 
    장음기호를 처리하여 음소 목록의 목록을 반환합니다.

    Args: 
        sep_phonemes (list[list[str]]): 문구 별 구분된 음소 목록 목록

    Returns: 
        list[list[str]]: 장음 기호를 처리한 음소 목록 목록 
    """
    
    for i in range(len(sep_phonemes)):
        if len(sep_phonemes[i]) == 0:
            continue
        if sep_phonemes[i][0] == "ー":
            if i != 0:
                prev_phoneme = sep_phonemes[i - 1][-1]
                if prev_phoneme in VOWELS:
                    sep_phonemes[i][0] = sep_phonemes[i - 1][-1]
                else:
                    sep_phonemes[i][0] = "-"
            else:
                sep_phonemes[i][0] = "-"
        if "ー" in sep_phonemes[i]:
            for j in range(len(sep_phonemes[i])):
                if sep_phonemes[i][j] == "ー":
                    sep_phonemes[i][j] = sep_phonemes[i][j - 1][-1]

    return sep_phonemes

__KATAKANA_PATTERN = re.compile(r"[\u30A0-\u30FF]+")
__MORA_PATTERN = re.compile(
    "|".join(
        map(re.escape, sorted(MORA_KATA_TO_MORA_PHONEMES.keys(), key=len, reverse=True))
    )
)
__LONG_PATTERN = re.compile(r"(\w)(ー*)")

def __kata_to_phoneme_list(text: str) -> list[str]:
    """ 
    가타카나 `text`를 받아 그대로 음소 기호 목록으로 변환합니다.
    주의 사항:
    - punctuation 또는 그 반복이 오면 punctuation들을 그대로 목록으로 반환합니다.
    - 시작에 이어지는 'ー'는 그대로 'ー'로 남겨둡니다. ('handle_long()'로 처리).
    - 문장의 '-'는 이전 음소 기호의 마지막 음소 기호로 변환됩니다.
    예: 
    'ーーソーナノカーー' -> ["ー", "ー", "s", "o", "o", "n", "a", "n", "o", "k", "a", "a", "a"]
    '?' -> ["?"]
    '!?!?!?!?!' -> ["!", "?", "!", "?", "!", "?", "!", "?", "!"]

    Args: 
        text (str): 가타카나 텍스트

    Returns: 
        list[str]: 음소 기호 목록 
    """
    if set(text).issubset(set(PUNCTUATIONS)):
        return list(text)
    if __KATAKANA_PATTERN.fullmatch(text) is None:
        raise ValueError(f"Input must be katakana only: {text}")
    
    def mora2phonemes(mora: str) -> str:
        consonant, vowel = MORA_KATA_TO_MORA_PHONEMES[mora]
        if consonant is None:
            return f" {vowel}"
        return f" {consonant} {vowel}"

    spaced_phonemes = __MORA_PATTERN.sub(lambda m: mora2phonemes(m.group()), text)
    
    long_replacement = lambda m: m.group(1) + (" " + m.group(1)) * len(m.group(2))
    spaced_phonemes = __LONG_PATTERN.sub(long_replacement, spaced_phonemes)
    
    return spaced_phonemes.strip().split(" ")

def __align_tones(
    phones_with_punct: list[str], phone_tone_list: list[tuple[str, int]]
) -> list[tuple[str, int]]:
    """
    예시: ...私は,, そう思う。
    phones_with_punct:
        [".", ".", ".", "w", "a", "t", "a", "sh", "i", "w", "a", ",", ",", "s", "o", "o", "o", "m", "o", "u", "."]
    phone_tone_list:
        [("w", 0), ("a", 0), ("t", 1), ("a", 1), ("sh", 1), ("i", 1), ("w", 1), ("a", 1), ("_", 0), ("s", 0), ("o", 0), ("o", 1), ("o", 1), ("m", 1), ("o", 1), ("u", 0))]
    Return:
        [(".", 0), (".", 0), (".", 0), ("w", 0), ("a", 0), ("t", 1), ("a", 1), ("sh", 1), ("i", 1), ("w", 1), ("a", 1), (",", 0), (",", 0), ("s", 0), ("o", 0), ("o", 1), ("o", 1), ("m", 1), ("o", 1), ("u", 0), (".", 0)]

    Args:
        phones_with_punct (list[str]): punctuation가 포함된 음소 리스트
        phone_tone_list (list[tuple[str, int]]): punctuation가 포함되지 않은 음소와 악센트의 쌍 리스트

    Returns:
        list[tuple[str, int]]: punctuation가 포함된 음소와 악센트의 쌍 리스트
    """

    result: list[tuple[str, int]] = []
    tone_index = 0
    for phone in phones_with_punct:
        if tone_index >= len(phone_tone_list):
            result.append((phone, 0))
        elif phone == phone_tone_list[tone_index][0]:
            result.append((phone, phone_tone_list[tone_index][1]))
            tone_index += 1
        elif phone in PUNCTUATIONS:
            result.append((phone, 0))
        else:
            logger.debug(f"phones: {phones_with_punct}")
            logger.debug(f"phone_tone_list: {phone_tone_list}")
            logger.debug(f"result: {result}")
            logger.debug(f"tone_index: {tone_index}")
            logger.debug(f"phone: {phone}")
            raise ValueError(f"Unexpected phone: {phone}")

    return result

def __distribute_phone(n_phone: int, n_word: int) -> list[int]:
    """
    왼쪽에서 오른쪽으로 1씩 분배하고, 다음에 다시 왼쪽에서 오른쪽으로 1씩 늘려가는 방식으로,
    음소의 수 `n_phone`를 단어의 수 `n_word`에 분배합니다.

    Args:
        n_phone (int): 음소의 수
        n_word (int): 단어의 수

    Returns:
        list[int]: 단어 별 음소의 수의 리스트
    """

    phones_per_word = [0] * n_word
    for _ in range(n_phone):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1

    return phones_per_word

class YomiError(Exception):
    """
    일본어 G2P 처리 중 발생하는 예외입니다.
    """