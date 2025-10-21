# ====================================================================
#  사용 예시: LLM Manager
# ====================================================================

from services.llm_manager import LLMManager

# ============================================
# 방법 1: 로컬 서버 모드 사용
# ============================================
print("=== 로컬 서버 모드 ===")
server_configs = {
    "luna": {
        "url": "http://localhost:8080",
        "alias": "luna-model"
    }
}

llm_server = LLMManager(mode="server", server_configs=server_configs)

response = llm_server.generate(
    target="luna",
    system_prompt="You are a helpful assistant.",
    user_prompt="Hello!",
    temperature=0.9,
    max_tokens=128
)

print("서버 응답:", response)


# ============================================
# 방법 2: Gemini API 모드 사용
# ============================================
print("\n=== Gemini API 모드 ===")
api_configs = {
    "gemini": {
        "api_key": "your-gemini-api-key",  # 또는 환경 변수 사용
        "model": "gemini-2.0-flash-exp"
    }
}

llm_api = LLMManager(mode="api", api_configs=api_configs)

response = llm_api.generate(
    target="gemini",
    system_prompt="You are a helpful assistant.",
    user_prompt="Hello!",
    model="gemini-2.0-flash-exp"  # 선택적으로 모델 지정
)

print("API 응답:", response)


# ============================================
# 방법 3: 설정 파일을 통한 동적 선택
# ============================================
print("\n=== 설정 파일 기반 선택 ===")

def create_llm_from_config(use_api=False):
    """설정에 따라 적절한 LLM 인스턴스 생성"""
    
    if use_api:
        # API 모드 (예: Gemini)
        api_configs = {
            "gemini": {
                "api_key": "your-api-key",
                "model": "gemini-2.0-flash-exp"
            }
        }
        return LLMManager(mode="api", api_configs=api_configs), "gemini"
    else:
        # 서버 모드 (로컬 LLM)
        server_configs = {
            "luna": {
                "url": "http://localhost:8080",
                "alias": "luna-model"
            }
        }
        return LLMManager(mode="server", server_configs=server_configs), "luna"


# 사용자 선택에 따라 LLM 생성
use_api_mode = True  # True: API 모드, False: 서버 모드

llm, target = create_llm_from_config(use_api=use_api_mode)

response = llm.generate(
    target=target,
    system_prompt="You are Luna, a helpful AI assistant.",
    user_prompt="안녕하세요!"
)

print(f"모드: {llm.get_mode()}")
print(f"사용 가능한 타겟: {llm.get_available_targets()}")
print(f"응답: {response}")


# ============================================
# 방법 4: 대화 내역을 사용한 생성
# ============================================
print("\n=== 대화 내역 사용 ===")

messages = [
    {"role": "system", "content": "You are Luna, a helpful AI assistant."},
    {"role": "user", "content": "안녕하세요!"},
    {"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요?"},
    {"role": "user", "content": "날씨가 어때요?"}
]

response = llm.generate(
    target=target,
    messages=messages
)

print("대화 응답:", response)
