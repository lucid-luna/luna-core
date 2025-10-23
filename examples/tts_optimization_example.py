# ====================================================================
#  File: examples/tts_optimization_example.py
# ====================================================================
"""
TTS 최적화 예제

이 예제는 L.U.N.A. 시스템의 TTS 최적화 기능을 보여줍니다:
- 오디오 캐싱
- 비동기 처리
- 병렬 세그먼트 처리
"""

import asyncio
import time
from services.tts import TTSService


def example_1_basic_sync():
    """예제 1: 기본 동기 TTS (캐싱 포함)"""
    print("\n" + "=" * 60)
    print("예제 1: 기본 TTS (동기)")
    print("=" * 60)
    
    tts = TTSService(device="cuda", enable_cache=True)
    
    text = "こんにちは、世界！"
    
    # 첫 번째 호출 (캐시 미스)
    print("\n첫 번째 호출...")
    start = time.time()
    result1 = tts.synthesize(text=text)
    time1 = time.time() - start
    
    print(f"소요 시간: {time1:.2f}초")
    print(f"캐시 사용: {result1.get('cached', False)}")
    print(f"오디오 URL: {result1['audio_url']}")
    
    # 두 번째 호출 (캐시 히트)
    print("\n두 번째 호출...")
    start = time.time()
    result2 = tts.synthesize(text=text)
    time2 = time.time() - start
    
    print(f"소요 시간: {time2:.2f}초")
    print(f"캐시 사용: {result2.get('cached', False)}")
    print(f"속도 향상: {time1/time2:.1f}배")


async def example_2_async():
    """예제 2: 비동기 TTS"""
    print("\n" + "=" * 60)
    print("예제 2: 비동기 TTS")
    print("=" * 60)
    
    tts = TTSService(device="cuda", enable_cache=True)
    
    texts = [
        "今日はいい天気ですね。",
        "お元気ですか？",
        "ありがとうございます。"
    ]
    
    # 순차 처리
    print("\n순차 처리...")
    start = time.time()
    for text in texts:
        result = await tts.synthesize_async(text=text)
        print(f"  완료: {text}")
    seq_time = time.time() - start
    print(f"소요 시간: {seq_time:.2f}초")
    
    # 병렬 처리
    print("\n병렬 처리...")
    start = time.time()
    tasks = [tts.synthesize_async(text=text) for text in texts]
    results = await asyncio.gather(*tasks)
    par_time = time.time() - start
    print(f"소요 시간: {par_time:.2f}초")
    print(f"속도 향상: {seq_time/par_time:.1f}배")


async def example_3_parallel_segments():
    """예제 3: 병렬 세그먼트 처리 (긴 텍스트)"""
    print("\n" + "=" * 60)
    print("예제 3: 병렬 세그먼트 처리")
    print("=" * 60)
    
    tts = TTSService(device="cuda", enable_cache=True)
    
    # 긴 텍스트 생성
    long_text = "これは非常に長いテキストです。" * 50
    print(f"\n텍스트 길이: {len(long_text)}자")
    
    # 일반 처리
    print("\n일반 처리...")
    start = time.time()
    result1 = await tts.synthesize_async(text=long_text)
    time1 = time.time() - start
    print(f"소요 시간: {time1:.2f}초")
    
    # 병렬 처리
    print("\n병렬 처리...")
    start = time.time()
    result2 = await tts.synthesize_parallel(
        text=long_text,
        max_parallel=4
    )
    time2 = time.time() - start
    print(f"소요 시간: {time2:.2f}초")
    print(f"세그먼트 수: {result2.get('segments', 0)}")
    print(f"속도 향상: {time1/time2:.1f}배")


def example_4_cache_management():
    """예제 4: 캐시 관리"""
    print("\n" + "=" * 60)
    print("예제 4: 캐시 관리")
    print("=" * 60)
    
    tts = TTSService(device="cuda", enable_cache=True)
    
    # 여러 번 호출
    texts = [
        "テスト1",
        "テスト2",
        "テスト3",
        "テスト1",  # 중복
        "テスト2",  # 중복
    ]
    
    print("\nTTS 호출 중...")
    for text in texts:
        result = tts.synthesize(text=text)
        cached = "캐시" if result.get('cached') else "생성"
        print(f"  {text}: {cached}")
    
    # 캐시 통계
    print("\n캐시 통계:")
    stats = tts.cache.get_stats()
    print(f"  캐시 크기: {stats['cache_size']}/{stats['max_cache_size']}")
    print(f"  총 요청: {stats['total_requests']}")
    print(f"  히트: {stats['hits']}")
    print(f"  미스: {stats['misses']}")
    print(f"  히트율: {stats['hit_rate']:.1f}%")
    
    # 캐시 정리
    print("\n캐시 정리...")
    removed = tts.cache.cleanup_expired()
    print(f"삭제된 항목: {removed}개")


async def example_5_emotion_optimization():
    """예제 5: 감정 분석 최적화"""
    print("\n" + "=" * 60)
    print("예제 5: 감정 분석 최적화")
    print("=" * 60)
    
    tts = TTSService(device="cuda", enable_cache=True)
    
    text = "こんにちは"
    
    # 감정 분석 포함 (기본)
    print("\n감정 분석 포함...")
    start = time.time()
    result1 = await tts.synthesize_async(
        text=text,
        use_emotion_analysis=True
    )
    time1 = time.time() - start
    print(f"소요 시간: {time1:.2f}초")
    print(f"감정: {result1.get('emotion')}")
    
    # 감정 분석 생략 (스타일 명시)
    print("\n감정 분석 생략...")
    start = time.time()
    result2 = await tts.synthesize_async(
        text=text,
        style="Neutral",
        style_weight=1.0,
        use_emotion_analysis=False
    )
    time2 = time.time() - start
    print(f"소요 시간: {time2:.2f}초")
    print(f"단축 시간: {(time1-time2)*1000:.0f}ms")


def example_6_api_requests():
    """예제 6: HTTP API 요청"""
    print("\n" + "=" * 60)
    print("예제 6: HTTP API 요청")
    print("=" * 60)
    
    import requests
    
    server_url = "http://localhost:8000"
    
    # 1. 기본 TTS
    print("\n1. 기본 TTS 요청...")
    response = requests.post(
        f"{server_url}/synthesize",
        json={
            "text": "こんにちは",
            "style": "Neutral",
            "style_weight": 1.0
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"  오디오 URL: {result['audio_url']}")
        print(f"  캐시 사용: {result.get('cached', False)}")
    else:
        print(f"  오류: {response.status_code}")
    
    # 2. 캐시 통계
    print("\n2. 캐시 통계...")
    response = requests.get(f"{server_url}/tts/cache/stats")
    if response.status_code == 200:
        stats = response.json()
        print(f"  히트율: {stats.get('hit_rate', 0):.1f}%")
        print(f"  캐시 크기: {stats.get('cache_size', 0)}")
    
    # 3. 캐시 정리
    print("\n3. 캐시 정리...")
    response = requests.post(f"{server_url}/tts/cache/cleanup")
    if response.status_code == 200:
        result = response.json()
        print(f"  {result.get('message', '')}")


if __name__ == "__main__":
    print("=" * 60)
    print("L.U.N.A. TTS 최적화 예제")
    print("=" * 60)
    
    # 예제 선택
    print("\n실행할 예제를 선택하세요:")
    print("1. 기본 TTS (캐싱)")
    print("2. 비동기 TTS")
    print("3. 병렬 세그먼트 처리")
    print("4. 캐시 관리")
    print("5. 감정 분석 최적화")
    print("6. HTTP API 요청")
    print("0. 모든 예제 실행")
    
    choice = input("\n선택 (0-6): ").strip()
    
    if choice == "1":
        example_1_basic_sync()
    elif choice == "2":
        asyncio.run(example_2_async())
    elif choice == "3":
        asyncio.run(example_3_parallel_segments())
    elif choice == "4":
        example_4_cache_management()
    elif choice == "5":
        asyncio.run(example_5_emotion_optimization())
    elif choice == "6":
        example_6_api_requests()
    elif choice == "0":
        print("\n모든 예제는 개별적으로 실행하세요.")
        print("(일부 예제는 TTS 모델이 필요합니다)")
    else:
        print("잘못된 선택입니다.")
    
    print("\n" + "=" * 60)
    print("예제 완료!")
    print("=" * 60)
