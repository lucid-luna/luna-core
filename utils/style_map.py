# ====================================================================
#  File: utils/style_map.py
# ====================================================================
"""
이 파일은 L.U.N.A.에서 각 감정 스타일에 대한 매핑을 정의합니다.
각 스타일은 고유한 ID와 이름을 가지며, 감정 분석 및 음성 합성에 사용됩니다.
"""

# ─────────────────────────────────────────────────────────────────────
# 감정 -> TTS 스타일 매핑
# ─────────────────────────────────────────────────────────────────────
EMOTION_TO_STYLE = {
    # 일반 감정
    "neutral": "Neutral",
    "something_else": "Neutral",
    "admiration": "Neutral",
    "curiosity": "Neutral",
    
    # 긍정 감정 = Happy
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
    
    # 부정 감정 = Sad
    "sadness": "Sad",
    "grief": "Sad",
    "disappointment": "Sad",
    
    # 분노 감정 = Angry
    "anger": "Angry",
    "annoyance": "Angry",
    "disapproval": "Angry",
    
    # 역겨움 감정 = Disgusted
    "disgust": "Disgusted",
    
    # 당황 감정 = Embarrassed
    "embarrassment": "Embarrassed",
    
    # 공포 감정 = Fearful
    "fear": "Fearful",
    "nervousness": "Fearful",
    
    # 놀람 감정 = Surprised
    "surprise": "Surprised",
    "realization": "Surprised",
    
    # 떨리는 감정 = Sexual
    "desire": "Sexual1",
    "confusion": "Sexual2",
}

# ─────────────────────────────────────────────────────────────────────
# TTS 스타일별 강도 설정
# ─────────────────────────────────────────────────────────────────────
STYLE_STRENGTHS = {
    "Neutral": 1.0,
    "Happy": 3.0,
    "Sad": 20.0,
    "Angry": 20.0,
    "Disgusted": 2.7,
    "Embarrassed": 3.7,
    "Fearful": 16.4,
    "Surprised": 20.0,
    "Sexual1": 1.5,
    "Sexual2": 2.0,
}

# ─────────────────────────────────────────────────────────────────────
# 감정 점수 임계값 (낮은 점수는 무시)
# ─────────────────────────────────────────────────────────────────────
EMOTION_THRESHOLD = 0.3  # 0.3 미만은 무시

# ─────────────────────────────────────────────────────────────────────
# 함수: 감정 -> 스타일 변환
# ─────────────────────────────────────────────────────────────────────
def get_style_from_emotion(emotion: str) -> tuple[str, float]:
    """
    감정 이름에 해당하는 TTS 스타일과 강도를 반환합니다.
    
    Args:
        emotion (str): 감정 이름 (예: "joy", "sadness")
        
    Returns:
        tuple[str, float]: (스타일 이름, 강도)
        
    Examples:
        >>> get_style_from_emotion("joy")
        ("Happy", 3.0)
        >>> get_style_from_emotion("unknown")
        ("Neutral", 1.0)
    """
    emotion_lower = emotion.lower().strip()
    style_name = EMOTION_TO_STYLE.get(emotion_lower, "Neutral")
    strength = STYLE_STRENGTHS.get(style_name, 1.0)
    
    print(f"[StyleMap] 감정 '{emotion}' -> 스타일 '{style_name}' (강도: {strength})")
    
    return style_name, strength

# ─────────────────────────────────────────────────────────────────────
# 함수: 최고 점수 감정 추출
# ─────────────────────────────────────────────────────────────────────
def get_top_emotion(emotion_scores: dict[str, float]) -> str | None:
    """
    감정 점수 딕셔너리에서 가장 높은 점수를 가진 감정을 반환합니다.
    
    Args:
        emotion_scores (dict[str, float]): {감정 이름: 점수} 형태의 딕셔너리
        
    Returns:
        str | None: 가장 높은 점수를 가진 감정 이름 (임계값 미만이면 None)
        
    Examples:
        >>> get_top_emotion({"joy": 0.8, "sadness": 0.2})
        "joy"
        >>> get_top_emotion({"joy": 0.1, "sadness": 0.05})
        None
    """
    if not emotion_scores:
        print("[StyleMap] 감정 점수가 비어있음 -> None 반환")
        return None
    
    # 최고 점수 감정 찾기
    top_emotion, top_score = max(emotion_scores.items(), key=lambda x: x[1])
    
    print(f"[StyleMap] 감정 분석 결과: {emotion_scores}")
    print(f"[StyleMap] 최고 점수 감정: '{top_emotion}' (점수: {top_score:.4f})")
    
    # 임계값 확인
    if top_score < EMOTION_THRESHOLD:
        print(f"[StyleMap] 점수 {top_score:.4f} < 임계값 {EMOTION_THRESHOLD} -> None 반환")
        return None
    
    return top_emotion

# ─────────────────────────────────────────────────────────────────────
# 함수: 감정 점수 -> 스타일 직접 변환
# ─────────────────────────────────────────────────────────────────────
def get_style_from_scores(emotion_scores: dict[str, float]) -> tuple[str, float]:
    """
    감정 점수 딕셔너리에서 직접 스타일과 강도를 반환합니다.
    
    Args:
        emotion_scores (dict[str, float]): {감정 이름: 점수} 형태의 딕셔너리
        
    Returns:
        tuple[str, float]: (스타일 이름, 강도)
        
    Examples:
        >>> get_style_from_scores({"joy": 0.8, "sadness": 0.2})
        ("Happy", 3.0)
        >>> get_style_from_scores({})
        ("Neutral", 1.0)
    """
    top_emotion = get_top_emotion(emotion_scores)
    
    if top_emotion is None:
        print("[StyleMap] 유효한 감정 없음 -> Neutral 사용")
        return "Neutral", 1.0
    
    return get_style_from_emotion(top_emotion)