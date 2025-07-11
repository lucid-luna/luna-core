# ====================================================================
#  File: models/nlp/japanese/user_dict/word_model.py
# ====================================================================
"""
このファイルは、VOICEVOX プロジェクトの VOICEVOX ENGINE からお借りしています。
引用元: https://github.com/VOICEVOX/voicevox_engine/blob/f181411ec69812296989d9cc583826c22eec87ae/voicevox_engine/model.py#L207
ライセンス: LGPL-3.0
詳しくは、このファイルと同じフォルダにある README.md を参照してください。
"""

from enum import Enum
from re import findall, fullmatch
from typing import List, Optional

from pydantic import BaseModel, Field, validator

USER_DICT_MIN_PRIORITY = 0
USER_DICT_MAX_PRIORITY = 10

class UserDictWord(BaseModel):
    """
    사전 컴파일에 사용되는 정보
    """

    surface: str = Field(title="표층형")
    priority: int = Field(
        title="우선도", ge=USER_DICT_MIN_PRIORITY, le=USER_DICT_MAX_PRIORITY
    )
    context_id: int = Field(title="문맥 ID", default=1348)
    part_of_speech: str = Field(title="품사")
    part_of_speech_detail_1: str = Field(title="품사 세부 분류 1")
    part_of_speech_detail_2: str = Field(title="품사 세부 분류 2")
    part_of_speech_detail_3: str = Field(title="품사 세부 분류 3")
    inflectional_type: str = Field(title="활용형")
    inflectional_form: str = Field(title="활용형")
    stem: str = Field(title="원형")
    yomi: str = Field(title="읽기")
    pronunciation: str = Field(title="발음")
    accent_type: int = Field(title="억양형")
    mora_count: Optional[int] = Field(title="모라 수", default=None)
    accent_associative_rule: str = Field(title="억양 결합 규칙")

    class Config:
        validate_assignment = True

    @validator("surface")
    def convert_to_zenkaku(cls, surface):
        return surface.translate(
            str.maketrans(
                "".join(chr(0x21 + i) for i in range(94)),
                "".join(chr(0xFF01 + i) for i in range(94)),
            )
        )

    @validator("pronunciation", pre=True)
    def check_is_katakana(cls, pronunciation):
        if not fullmatch(r"[ァ-ヴー]+", pronunciation):
            raise ValueError("[L.U.N.A.] 발음은 유효한 카타카나여야 합니다.")
        sutegana = ["ァ", "ィ", "ゥ", "ェ", "ォ", "ャ", "ュ", "ョ", "ヮ", "ッ"]
        for i in range(len(pronunciation)):
            if pronunciation[i] in sutegana:
                if i < len(pronunciation) - 1 and (
                    pronunciation[i + 1] in sutegana[:-1]
                    or (
                        pronunciation[i] == sutegana[-1]
                        and pronunciation[i + 1] == sutegana[-1]
                    )
                ):
                    raise ValueError("[L.U.N.A.] 유효하지 않은 발음입니다.")
            if pronunciation[i] == "ヮ":
                if i != 0 and pronunciation[i - 1] not in ["ク", "グ"]:
                    raise ValueError(
                        "[L.U.N.A.] 무효한 발음입니다.(「くゎ」「ぐゎ」이외의「ゎ」의 사용)"
                    )
        return pronunciation

    @validator("mora_count", pre=True, always=True)
    def check_mora_count_and_accent_type(cls, mora_count, values):
        if "pronunciation" not in values or "accent_type" not in values:
            return mora_count

        if mora_count is None:
            rule_others = (
                "[イ][ェ]|[ヴ][ャュョ]|[トド][ゥ]|[テデ][ィャュョ]|[デ][ェ]|[クグ][ヮ]"
            )
            rule_line_i = "[キシチニヒミリギジビピ][ェャュョ]"
            rule_line_u = "[ツフヴ][ァ]|[ウスツフヴズ][ィ]|[ウツフヴ][ェォ]"
            rule_one_mora = "[ァ-ヴー]"
            mora_count = len(
                findall(
                    f"(?:{rule_others}|{rule_line_i}|{rule_line_u}|{rule_one_mora})",
                    values["pronunciation"],
                )
            )

        if not 0 <= values["accent_type"] <= mora_count:
            raise ValueError(
                "[L.U.N.A.] 잘못된 억양형입니다. ({})。 expect: 0 <= accent_type <= {}".format(
                    values["accent_type"], mora_count
                )
            )
        return mora_count


class PartOfSpeechDetail(BaseModel):
    """
    품사별 정보
    """

    part_of_speech: str = Field(title="품사")
    part_of_speech_detail_1: str = Field(title="품사 세부 분류 1")
    part_of_speech_detail_2: str = Field(title="품사 세부 분류 2")
    part_of_speech_detail_3: str = Field(title="품사 세부 분류 3")
    context_id: int = Field(title="문맥 ID")
    cost_candidates: List[int] = Field(title="비용의 퍼센타일")
    accent_associative_rules: List[str] = Field(title="억양 결합 규칙의 목록")


class WordTypes(str, Enum):
    """
    FastAPI에서 word_type 인자를 검증할 때 사용하는 클래스
    """

    PROPER_NOUN = "PROPER_NOUN"
    COMMON_NOUN = "COMMON_NOUN"
    VERB = "VERB"
    ADJECTIVE = "ADJECTIVE"
    SUFFIX = "SUFFIX"
