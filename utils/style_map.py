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
    "curiosity": "Sexual2",
    "confusion": "Sexual2",
}

STYLE_STRENGTHS = {
    # 스타일 이름: 강도 (0.0 ~ 20.0)
    "Neutral": 1.0,
    "Happy": 1.0,
    "Sad": 1.0,
    "Angry": 1.0,
    "Disgusted": 1.0,
    "Embarrassed": 1.0,
    "Fearful": 1.0,
    "Surprised": 1.0,
    "Sexual1": 1.0,
    "Sexual2": 1.0,
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