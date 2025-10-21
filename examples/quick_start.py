# ====================================================================
#  간단한 사용 예시 - LLM Manager 빠른 시작
# ====================================================================

from utils.llm_config import get_llm_manager

# ============================================
# 방법 1: 자동 모드 (환경 변수 또는 설정 파일 기반)
# ============================================
print("=== 자동 모드 ===")
llm, target = get_llm_manager(auto_mode=True)

if llm and target:
    response = llm.generate(
        target=target,
        system_prompt="You are Luna, a helpful AI assistant.",
        user_prompt="안녕하세요!"
    )
    
    # 응답 추출
    if "choices" in response:
        content = response["choices"][0]["message"]["content"]
        print(f"Luna: {content}")
    else:
        print(f"오류: {response}")


# ============================================
# 방법 2: 대화형 모드 (사용자가 선택)
# ============================================
print("\n=== 대화형 모드 ===")
llm, target = get_llm_manager(auto_mode=False)

if llm and target:
    print(f"\n현재 모드: {llm.get_mode()}")
    print(f"사용 가능한 타겟: {llm.get_available_targets()}")
    print(f"선택된 타겟: {target}")
    
    # 대화 시작
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["exit", "quit", "종료"]:
            print("종료합니다.")
            break
        
        if not user_input:
            continue
        
        response = llm.generate(
            target=target,
            system_prompt="You are Luna, a helpful AI assistant. Answer in Korean.",
            user_prompt=user_input
        )
        
        if "choices" in response:
            content = response["choices"][0]["message"]["content"]
            print(f"Luna: {content}")
        else:
            print(f"오류: {response}")
