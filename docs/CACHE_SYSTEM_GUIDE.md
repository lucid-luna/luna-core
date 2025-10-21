# 응답 캐싱 시스템 가이드

## ✅ 1단계 완료: 응답 캐싱 시스템

API 비용을 절감하고 응답 속도를 향상시키는 지능형 캐싱 시스템이 추가되었습니다!

## 주요 기능

### 1. 정확한 매칭
- 동일한 질문에 대해 캐시된 응답 즉시 반환
- API 호출 없이 0ms 응답

### 2. 유사도 매칭
- 비슷한 질문도 인식하여 캐시 활용
- 예: "안녕" ≈ "안녕하세요" (유사도 85% 이상)

### 3. 컨텍스트 인식
- 대화 히스토리를 고려한 캐싱
- 같은 질문이라도 대화 맥락에 따라 다른 응답

### 4. 자동 만료
- TTL (Time To Live) 기반 자동 만료
- 기본값: 1시간 (3600초)

### 5. 크기 제한
- 최대 100개 캐시 항목 유지
- 오래된 항목부터 자동 삭제

## 설정

### 기본 설정 (자동 활성화)

```python
# main.py에서 자동으로 활성화됨
llm_service = LLMManager(
    mode="api",
    api_configs={
        "gemini": {
            "api_key": "your-key",
            "model": "gemini-2.5-flash"
        }
    }
)
# 캐싱이 자동으로 활성화됩니다!
```

### 캐시 비활성화

```python
from services.llm_api import LLMAPIService

api_service = LLMAPIService(
    api_configs=api_configs,
    enable_cache=False  # 캐싱 비활성화
)
```

### 캐시 설정 커스터마이징

```python
api_service = LLMAPIService(
    api_configs=api_configs,
    enable_cache=True,
    cache_ttl=7200  # 2시간으로 변경
)
```

## API 엔드포인트

### 1. 캐시 통계 조회

**GET** `/cache/stats`

```bash
curl http://localhost:8000/cache/stats
```

응답 예시:
```json
{
  "hits": 45,
  "misses": 23,
  "saves": 23,
  "hit_rate": "66.2%",
  "total_cached": 23,
  "max_size": 100,
  "ttl_seconds": 3600
}
```

### 2. 만료된 캐시 정리

**POST** `/cache/cleanup`

```bash
curl -X POST http://localhost:8000/cache/cleanup
```

응답:
```json
{
  "status": "success",
  "removed": 5,
  "message": "5개의 만료된 캐시가 삭제되었습니다."
}
```

### 3. 모든 캐시 삭제

**DELETE** `/cache/clear`

```bash
curl -X DELETE http://localhost:8000/cache/clear
```

응답:
```json
{
  "status": "success",
  "message": "모든 캐시가 삭제되었습니다."
}
```

## 효과

### 비용 절감
```
캐시 히트율 66% 가정 시:
- 100번 요청 → 34번만 실제 API 호출
- API 비용 66% 절감! 💰
```

### 속도 향상
```
캐시 히트 시:
- API 호출: 500-2000ms
- 캐시 조회: <5ms
- 최대 400배 빠른 응답! ⚡
```

## 동작 원리

### 캐시 키 생성

```python
# 입력: "안녕하세요"
# 정규화: "안녕하세요" → "안녕하세요" (소문자, 공백 제거)
# 해시: SHA256("안녕하세요|gemini-2.5-flash|context_hash")
# 결과: "a1b2c3d4..."
```

### 유사도 계산

```python
# 단어 기반 Jaccard 유사도
text1 = "안녕하세요 반가워요"
text2 = "안녕 반갑습니다"

words1 = {"안녕하세요", "반가워요"}
words2 = {"안녕", "반갑습니다"}

# 교집합 / 합집합
similarity = 0 / 4 = 0.0  # (공통 단어 없음)

# 하지만 실제로는 형태소 분석으로 개선 가능
```

## 로그 확인

### 캐시 히트 (정확 일치)
```
[Cache] 캐시 히트 (정확 일치): 안녕하세요...
[L.U.N.A. LLM API] (API 호출 없음)
```

### 캐시 히트 (유사 매칭)
```
[Cache] 캐시 히트 (유사도: 0.89): 안녕...
[L.U.N.A. LLM API] (API 호출 없음)
```

### 캐시 미스
```
[Cache] 캐시 미스: 오늘 날씨 어때?...
[L.U.N.A. LLM API] Gemini API 호출 중...
[Cache] 캐시 저장: 오늘 날씨 어때?...
```

## 캐시 파일 구조

`./cache/response_cache.json`:

```json
{
  "a1b2c3d4...": {
    "prompt": "안녕하세요",
    "response": {
      "choices": [{
        "message": {
          "role": "assistant",
          "content": "음, 다엘. 내 생각 하고 있었어?"
        }
      }]
    },
    "model": "gemini-2.5-flash",
    "timestamp": 1729502400.123
  }
}
```

## 최적화 팁

### 1. 캐시 히트율 높이기
```python
# 유사도 임계값 조정 (cache.py)
similarity_threshold=0.80  # 더 느슨하게 (기본: 0.85)
```

### 2. 캐시 크기 늘리기
```python
# 더 많은 항목 저장
max_cache_size=200  # (기본: 100)
```

### 3. TTL 조정
```python
# 더 오래 유지
cache_ttl=86400  # 24시간 (기본: 3600초)
```

## 주의사항

1. **개인정보**: 민감한 대화는 캐시 비활성화 권장
2. **디스크 공간**: 캐시 파일 주기적 확인
3. **실시간 정보**: 날씨/뉴스 등은 TTL 짧게 설정
4. **컨텍스트 변화**: 대화 맥락이 바뀌면 새로운 캐시 생성

## 다음 단계

✅ **1단계 완료**: 응답 캐싱 시스템
🔄 **2단계 진행 중**: 스트리밍 응답 구현 예정

준비되면 다음 기능을 추가할 수 있습니다!
