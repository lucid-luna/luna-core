# ====================================================================
#  File: examples/streaming_example.py
# ====================================================================
"""
LLM 스트리밍 응답 예제

이 예제는 L.U.N.A. 시스템에서 스트리밍 응답을 사용하는 방법을 보여줍니다.
"""

import requests
import json
import sseclient  # pip install sseclient-py


def stream_interaction_api(user_input: str, server_url: str = "http://localhost:8000"):
    """
    API를 통해 스트리밍 방식으로 응답을 받습니다.
    
    Args:
        user_input: 사용자 입력
        server_url: L.U.N.A. 서버 URL
    """
    url = f"{server_url}/interact/stream"
    headers = {"Content-Type": "application/json"}
    data = {"input": user_input}
    
    print(f"\n[사용자] {user_input}")
    print("[L.U.N.A.] ", end="", flush=True)
    
    try:
        response = requests.post(url, json=data, headers=headers, stream=True)
        response.raise_for_status()
        
        client = sseclient.SSEClient(response)
        
        for event in client.events():
            if event.data:
                try:
                    data = json.loads(event.data)
                    event_type = data.get("type")
                    content = data.get("data")
                    
                    if event_type == "emotion":
                        print(f"\n[감정 분석] {content}")
                        print("[L.U.N.A.] ", end="", flush=True)
                    
                    elif event_type == "translation":
                        print(f"\n[번역됨] {content}")
                        print("[L.U.N.A.] ", end="", flush=True)
                    
                    elif event_type == "llm_chunk":
                        print(content, end="", flush=True)
                    
                    elif event_type == "complete":
                        print("\n[완료]")
                    
                    elif event_type == "error":
                        print(f"\n[오류] {content}")
                        break
                
                except json.JSONDecodeError:
                    continue
    
    except requests.exceptions.RequestException as e:
        print(f"\n[오류] API 요청 실패: {e}")


def stream_with_llm_manager():
    """
    LLMManager를 직접 사용하여 스트리밍 응답을 받습니다.
    """
    from services.llm_manager import LLMManager
    
    # API 모드로 초기화
    llm_manager = LLMManager(
        mode="api",
        api_configs={
            "gemini": {
                "model": "gemini-2.5-flash"
            }
        }
    )
    
    # 스트리밍 요청
    messages = [
        {"role": "system", "content": "당신은 친절한 AI 어시스턴트입니다."},
        {"role": "user", "content": "인공지능의 미래에 대해 설명해주세요."}
    ]
    
    print("\n[L.U.N.A.] ", end="", flush=True)
    
    stream_response = llm_manager.generate(
        target="gemini",
        messages=messages,
        stream=True
    )
    
    full_response = ""
    for chunk in stream_response:
        if "error" in chunk:
            print(f"\n[오류] {chunk['error']}")
            break
        
        if chunk.get("choices") and len(chunk["choices"]) > 0:
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")
            
            if content:
                print(content, end="", flush=True)
                full_response += content
            
            finish_reason = chunk["choices"][0].get("finish_reason")
            if finish_reason == "stop":
                print("\n")
                break
    
    return full_response


def stream_with_llm_api():
    """
    LLMAPIService를 직접 사용하여 스트리밍 응답을 받습니다.
    """
    from services.llm_api import LLMAPIService
    
    # API 서비스 초기화
    api_service = LLMAPIService(
        api_configs={
            "gemini": {
                "model": "gemini-2.5-flash"
            }
        }
    )
    
    # 스트리밍 요청
    print("\n[L.U.N.A.] ", end="", flush=True)
    
    stream_response = api_service.generate(
        target_provider="gemini",
        system_prompt="당신은 친절한 AI 어시스턴트입니다.",
        user_prompt="파이썬의 주요 특징을 간단히 설명해주세요.",
        stream=True
    )
    
    full_response = ""
    for chunk in stream_response:
        if "error" in chunk:
            print(f"\n[오류] {chunk['error']}")
            break
        
        if chunk.get("choices") and len(chunk["choices"]) > 0:
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")
            
            if content:
                print(content, end="", flush=True)
                full_response += content
            
            finish_reason = chunk["choices"][0].get("finish_reason")
            if finish_reason == "stop":
                print("\n")
                break
    
    return full_response


if __name__ == "__main__":
    print("=" * 60)
    print("L.U.N.A. 스트리밍 응답 예제")
    print("=" * 60)
    
    # 예제 1: HTTP API를 통한 스트리밍
    print("\n[예제 1] HTTP API 스트리밍")
    print("-" * 60)
    stream_interaction_api("안녕하세요! 오늘 날씨가 어때요?")
    
    # 예제 2: LLMManager를 사용한 스트리밍
    print("\n[예제 2] LLMManager 스트리밍")
    print("-" * 60)
    stream_with_llm_manager()
    
    # 예제 3: LLMAPIService를 사용한 스트리밍
    print("\n[예제 3] LLMAPIService 스트리밍")
    print("-" * 60)
    stream_with_llm_api()
    
    print("\n" + "=" * 60)
    print("모든 예제 완료!")
    print("=" * 60)
