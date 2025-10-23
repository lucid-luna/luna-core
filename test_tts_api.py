"""TTS API 테스트 스크립트"""
import requests
import json

API_URL = "http://localhost:8000"

def test_tts(text: str, output_file: str = "test_output.wav"):
    """TTS 합성 테스트"""
    print("=== L.U.N.A. TTS API 테스트 ===\n")
    
    # 1. 서버 상태 확인
    print("[1] 서버 상태 확인...")
    try:
        response = requests.get(f"{API_URL}/health")
        print("✓ 서버 정상 작동\n")
    except Exception as e:
        print(f"✗ 서버 접속 실패: {e}")
        return
    
    # 2. TTS 합성 요청
    print(f"[2] TTS 합성 요청 중...")
    print(f"  텍스트: {text}")
    
    payload = {
        "text": text,
        "model_name": "Luna",
        "language": "JP",
        "style": "Neutral",
        "style_weight": 1.0
    }
    
    try:
        response = requests.post(
            f"{API_URL}/synthesize",  # 올바른 경로
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            # JSON 응답에서 오디오 URL 추출
            result = response.json()
            audio_url = result.get("audio_url")
            
            if audio_url:
                # 실제 오디오 파일 다운로드
                audio_response = requests.get(f"{API_URL}{audio_url}")
                
                if audio_response.status_code == 200:
                    with open(output_file, "wb") as f:
                        f.write(audio_response.content)
                    
                    print(f"✓ TTS 합성 완료!")
                    print(f"  스타일: {result.get('style')}")
                    print(f"  캐시됨: {result.get('cached')}")
                    print(f"  파일 저장됨: {output_file}")
                    print(f"  파일 크기: {len(audio_response.content):,} bytes")
                    
                    # Windows에서 자동 재생
                    import os
                    if os.name == 'nt':
                        print(f"\n[3] 음성 파일 재생...")
                        os.system(f'start {output_file}')
                else:
                    print(f"✗ 오디오 다운로드 실패: {audio_response.status_code}")
            else:
                print(f"✗ 응답에 audio_url이 없습니다: {result}")
        else:
            print(f"✗ TTS 합성 실패: {response.status_code}")
            print(f"  에러: {response.text}")
            
    except Exception as e:
        print(f"✗ 요청 실패: {e}")


if __name__ == "__main__":
    # 테스트할 텍스트 (일본어)
    test_text = "こんにちは、ルナです。テストを開始します。"
    
    # 한국어 테스트 (한국어 지원 시)
    # test_text = "안녕하세요, 루나입니다. 테스트를 시작합니다."
    
    test_tts(test_text)
