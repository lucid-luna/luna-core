# ====================================================================
#  File: utils/style_map.py
# ====================================================================
"""
이 파일은 L.U.N.A.에서 각 감정 스타일에 대한 매핑을 정의합니다.
각 스타일은 고유한 ID와 이름을 가지며, 감정 분석 및 음성 합성에 사용됩니다.
"""

EMOTION_TO_STYLE = {
    # 감정 ID: (감정 이름, 스타일 이름)
    # 일반 감정
    "neutral": "Neutral",
    "something_else": "Neutral",
    "admiration": "Neutral",
    "curiosity": "Neutral",  # 질문 관련 감정은 전부 중립 처리
    
    # 긍정 감정 = happy
    "joy": "Happy",
    "amusement": "Happy",
    "gratitude": "Happy",
    "love": "Happy",
    "excitement": "Happy",
    "optimism": "Happy",
    "pride": "Happy",
    "approval": "Happy",
    "caring": "Happy",
    "relief": "Happy",
    
    # 부정 감정 = sad
    "sadness": "Sad",
    "grief": "Sad",
    "disappointment": "Sad",
    
    # 분노 감정 = angry
    "anger": "Angry",
    "annoyance": "Angry",
    "disapproval": "Angry",
    
    # 역겨움 감정 = disgusted
    "disgust": "Disgusted",
    
    # 당황 감정 = embarrassed
    "embarrassment": "Embarrassed",
    
    # 공포 감정 = fearful
    "fear": "Fearful",
    "nervousness": "Fearful",
    
    # 놀람 감정 = surprised
    "surprise": "Surprised",
    "realization": "Surprised",
    
    # 떨리는 감정 => Sexual1/2 사용
    "desire": "Sexual1",
    "confusion": "Sexual2",
}

STYLE_STRENGTHS = {
    # 스타일 이름: 강도 (0.0 ~ 20.0)
    "Neutral": 1.0, # 기본 스타일, 감정이 없거나 중립적인 상태
    "Happy": 3.0, # Neutral과 비슷하지만, 행복감이 약간 더 높음
    "Sad": 20.0, # 20이 제일 슬픔이 정확함
    "Angry": 20.0, # 20이 제일 분노가 정확함
    "Disgusted": 2.7, # 3 이상으로 넘어가지 않게 주의
    "Embarrassed": 3.7, # 4 이상으로 넘어가지 않게 주의
    "Fearful": 16.4, # 일정 이상 넘어갈 시 공포감이 강해짐
    "Surprised": 20.0, # Neutral과 비슷하지만, 놀람의 강도가 약간 더 높음
    "Sexual1": 1.5, # 강도는 낮지만, 감정 표현이 강함 ( 떨리는 목소리 )
    "Sexual2": 2.0, # 감정 표현이 강함 ( 흐느낀 뒤 목소리 혹은 떨리는 목소리 )
}

def get_style_from_emotion(emotion: str) -> tuple[str, float]:
    """
    감정 이름에 해당하는 스타일과 강도를 반환합니다.
    
    Args:
        emotion (str): 감정 이름 (예: "joy", "sadness")
        
    Returns:
        tuple[str, float]: (스타일 이름, 강도)
    """
    style_name = EMOTION_TO_STYLE.get(emotion, "Neutral")
    strength = STYLE_STRENGTHS.get(style_name, 1.0)
    return style_name, strength

def get_top_emotion(emotion_scores: dict[str, float]) -> str:
    """
    감정 점수 딕셔너리에서 가장 높은 점수를 가진 감정을 반환합니다.
    
    Args:
        emotion_scores (dict[str, float]): {감정 이름: 점수} 형태의 딕셔너리
        
    Returns:
        str: 가장 높은 점수를 가진 감정 이름
    """
    if not emotion_scores:
        return "neutral"
    
    return max(emotion_scores.items(), key=lambda x: x[1])[0]