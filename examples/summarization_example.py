# ====================================================================
#  File: examples/summarization_example.py
# ====================================================================
"""
대화 요약 시스템 예제

이 예제는 L.U.N.A. 시스템에서 대화 요약 기능을 사용하는 방법을 보여줍니다.
"""

import requests
import time
from services.memory import MemoryService
from services.llm_manager import LLMManager


def example_1_automatic_summarization():
    """예제 1: 자동 요약 - 20턴 도달 시 자동 실행"""
    print("\n" + "=" * 60)
    print("예제 1: 자동 요약")
    print("=" * 60)
    
    # LLM 서비스 초기화
    llm = LLMManager(
        mode="api",
        api_configs={"gemini": {"model": "gemini-2.5-flash"}}
    )
    
    # 메모리 서비스 초기화 (자동 요약 활성화)
    memory = MemoryService(
        memory_dir="./memory_test",
        max_entries=50,
        max_context_turns=6,
        summary_threshold=10,  # 테스트를 위해 10턴으로 설정
        enable_auto_summary=True,
        llm_service=llm
    )
    
    # 대화 시뮬레이션
    conversations = [
        ("파이썬 배우고 싶어요", "파이썬은 초보자에게 좋은 언어입니다. 무엇부터 시작하시겠어요?"),
        ("변수 선언은 어떻게 하나요?", "파이썬에서는 x = 10처럼 간단하게 선언합니다."),
        ("함수는요?", "def 키워드를 사용합니다. def my_func(): 형식입니다."),
        ("리스트 사용법 알려주세요", "리스트는 []로 만듭니다. my_list = [1, 2, 3]처럼요."),
        ("반복문은?", "for item in my_list: 형식으로 사용합니다."),
        ("딕셔너리는 뭔가요?", "키-값 쌍으로 저장하는 자료구조입니다. {'name': 'John'}처럼요."),
        ("클래스 만들기", "class MyClass: 형식으로 시작합니다."),
        ("상속은 어떻게?", "class Child(Parent): 형식으로 상속받습니다."),
        ("예외 처리는?", "try-except 블록을 사용합니다."),
        ("파일 읽기는?", "with open('file.txt', 'r') as f: 형식입니다."),
        ("이제 pandas 배우고 싶어요", "좋습니다! pandas는 데이터 분석에 유용합니다."),
    ]
    
    print("\n대화 시작...")
    for i, (user, assistant) in enumerate(conversations, 1):
        print(f"\n턴 {i}:")
        print(f"  사용자: {user}")
        print(f"  어시스턴트: {assistant}")
        
        memory.add_entry(user, assistant)
        
        # 10턴 도달 시 자동 요약 실행됨
        if i == 10:
            print("\n  ⚡ 10턴 도달 → 자동 요약 실행!")
            time.sleep(2)  # 요약 생성 대기
    
    # 요약 확인
    print("\n" + "-" * 60)
    print("생성된 요약:")
    print("-" * 60)
    summary = memory.get_summary()
    if summary:
        print(summary)
    else:
        print("요약이 생성되지 않았습니다.")
    
    # 메모리 상태 확인
    print("\n" + "-" * 60)
    print("메모리 상태:")
    print("-" * 60)
    stats = memory.get_memory_stats()
    print(f"총 항목: {stats['total_entries']}")
    print(f"대화: {stats['conversations']}")
    print(f"요약: {stats['summaries']}")


def example_2_manual_summarization():
    """예제 2: 수동 요약 - API를 통한 요약 요청"""
    print("\n" + "=" * 60)
    print("예제 2: 수동 요약 (API)")
    print("=" * 60)
    
    server_url = "http://localhost:8000"
    
    # 1. 대화 추가
    print("\n대화 추가 중...")
    for i in range(5):
        response = requests.post(
            f"{server_url}/interact",
            json={"input": f"테스트 질문 {i+1}"}
        )
        print(f"  턴 {i+1} 완료")
    
    # 2. 수동 요약 실행
    print("\n수동 요약 실행...")
    response = requests.post(f"{server_url}/memory/summarize")
    
    if response.status_code == 200:
        result = response.json()
        print(f"상태: {result['status']}")
        print(f"메시지: {result['message']}")
        if result.get('summary'):
            print(f"\n요약:\n{result['summary']}")
    else:
        print(f"오류: {response.status_code}")


def example_3_context_with_summary():
    """예제 3: 요약이 포함된 컨텍스트 확인"""
    print("\n" + "=" * 60)
    print("예제 3: 요약이 포함된 컨텍스트")
    print("=" * 60)
    
    llm = LLMManager(
        mode="api",
        api_configs={"gemini": {"model": "gemini-2.5-flash"}}
    )
    
    memory = MemoryService(
        memory_dir="./memory_test",
        max_entries=50,
        max_context_turns=3,
        summary_threshold=5,
        enable_auto_summary=True,
        llm_service=llm
    )
    
    # 대화 추가 (5턴 이상)
    print("\n대화 추가 중...")
    for i in range(8):
        memory.add_entry(
            f"질문 {i+1}",
            f"답변 {i+1}"
        )
    
    # 컨텍스트 확인
    print("\n" + "-" * 60)
    print("LLM에 전달될 컨텍스트:")
    print("-" * 60)
    
    context = memory.get_context_for_llm()
    for i, msg in enumerate(context, 1):
        role = msg['role']
        content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
        print(f"{i}. [{role}] {content}")
    
    print(f"\n총 메시지 수: {len(context)}")


def example_4_api_endpoints():
    """예제 4: API 엔드포인트 사용"""
    print("\n" + "=" * 60)
    print("예제 4: API 엔드포인트")
    print("=" * 60)
    
    server_url = "http://localhost:8000"
    
    # 1. 메모리 통계
    print("\n1. 메모리 통계 조회")
    print("-" * 40)
    response = requests.get(f"{server_url}/memory/stats")
    if response.status_code == 200:
        stats = response.json()
        print(f"총 항목: {stats.get('total_entries', 0)}")
        print(f"대화: {stats.get('conversations', 0)}")
        print(f"요약: {stats.get('summaries', 0)}")
        print(f"자동 요약: {stats.get('auto_summary_enabled', False)}")
        print(f"요약 임계값: {stats.get('summary_threshold', 0)}턴")
    
    # 2. 요약 조회
    print("\n2. 현재 요약 조회")
    print("-" * 40)
    response = requests.get(f"{server_url}/memory/summary")
    if response.status_code == 200:
        result = response.json()
        if result.get('summary'):
            print(result['summary'])
        else:
            print(result.get('message', '요약 없음'))
    
    # 3. 최근 대화 조회
    print("\n3. 최근 대화 조회 (5개)")
    print("-" * 40)
    response = requests.get(f"{server_url}/memory/recent?count=5")
    if response.status_code == 200:
        result = response.json()
        recent = result.get('recent_conversations', [])
        for i, conv in enumerate(recent, 1):
            print(f"{i}. {conv.get('type', 'conversation')}")
            if conv.get('type') == 'summary':
                content = conv.get('content', '')[:50] + "..."
                print(f"   요약: {content}")
            else:
                user = conv.get('user', '')[:30] + "..."
                print(f"   사용자: {user}")


def example_5_cost_comparison():
    """예제 5: 비용 비교 시뮬레이션"""
    print("\n" + "=" * 60)
    print("예제 5: 요약 사용 시 비용 절감 효과")
    print("=" * 60)
    
    # 가정
    turns_per_conversation = 100
    avg_tokens_per_turn = 100
    summary_cost = 700  # 요약 생성 1회 비용 (토큰)
    summary_result_tokens = 200  # 요약 결과 크기
    recent_turns_kept = 6
    
    # 요약 없이
    without_summary_tokens = turns_per_conversation * avg_tokens_per_turn
    
    # 요약 사용
    with_summary_tokens = summary_cost + (summary_result_tokens + recent_turns_kept * avg_tokens_per_turn)
    
    # 100회 대화 시
    conversations = 100
    without_summary_total = without_summary_tokens * conversations
    with_summary_total = summary_cost + (summary_result_tokens + recent_turns_kept * avg_tokens_per_turn) * conversations
    
    print(f"\n[단일 대화 비교]")
    print(f"  요약 없음: {without_summary_tokens:,} 토큰")
    print(f"  요약 사용: {with_summary_tokens:,} 토큰")
    print(f"  절감율: {((without_summary_tokens - with_summary_tokens) / without_summary_tokens * 100):.1f}%")
    
    print(f"\n[100회 대화 비교]")
    print(f"  요약 없음: {without_summary_total:,} 토큰")
    print(f"  요약 사용: {with_summary_total:,} 토큰")
    print(f"  절감율: {((without_summary_total - with_summary_total) / without_summary_total * 100):.1f}%")
    
    # 비용 계산 (Gemini 2.5 Flash 기준)
    cost_per_million = 0.075  # $0.075 per 1M tokens
    
    without_summary_cost = (without_summary_total / 1_000_000) * cost_per_million
    with_summary_cost = (with_summary_total / 1_000_000) * cost_per_million
    
    print(f"\n[예상 비용 (Gemini 2.5 Flash)]")
    print(f"  요약 없음: ${without_summary_cost:.4f}")
    print(f"  요약 사용: ${with_summary_cost:.4f}")
    print(f"  절감액: ${without_summary_cost - with_summary_cost:.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("L.U.N.A. 대화 요약 시스템 예제")
    print("=" * 60)
    
    # 예제 선택
    print("\n실행할 예제를 선택하세요:")
    print("1. 자동 요약 (메모리 서비스 직접 사용)")
    print("2. 수동 요약 (API 사용)")
    print("3. 요약이 포함된 컨텍스트 확인")
    print("4. API 엔드포인트 테스트")
    print("5. 비용 절감 효과 시뮬레이션")
    print("0. 모든 예제 실행")
    
    choice = input("\n선택 (0-5): ").strip()
    
    if choice == "1":
        example_1_automatic_summarization()
    elif choice == "2":
        example_2_manual_summarization()
    elif choice == "3":
        example_3_context_with_summary()
    elif choice == "4":
        example_4_api_endpoints()
    elif choice == "5":
        example_5_cost_comparison()
    elif choice == "0":
        example_5_cost_comparison()
        # example_1_automatic_summarization()  # 주석 처리: 실제 LLM 호출 필요
        print("\n나머지 예제는 주석을 해제하고 실행하세요.")
    else:
        print("잘못된 선택입니다.")
    
    print("\n" + "=" * 60)
    print("예제 완료!")
    print("=" * 60)
