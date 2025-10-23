"""uvicorn 없이 직접 FastAPI 서버 실행"""
import torch
import os

# BERT 로딩 건너뛰기 (임시)
os.environ["SKIP_TTS_WARMUP"] = "1"

print("=" * 80)
print("XPU 상태 확인 (uvicorn 없이)")
print("=" * 80)

print(f"\n[1] torch.xpu 상태:")
print(f"  hasattr(torch, 'xpu'): {hasattr(torch, 'xpu')}")
if hasattr(torch, 'xpu'):
    print(f"  torch.xpu.is_available(): {torch.xpu.is_available()}")
    print(f"  torch.xpu.device_count(): {torch.xpu.device_count()}")

print(f"\n[2] main.py import 시작...")
import main

print(f"\n[3] main.py import 후 XPU 상태:")
if hasattr(torch, 'xpu'):
    print(f"  torch.xpu.is_available(): {torch.xpu.is_available()}")
    print(f"  torch.xpu.device_count(): {torch.xpu.device_count()}")

print(f"\n[4] 직접 FastAPI 서버 실행...")
import uvicorn

# uvicorn 직접 실행
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("서버 시작 (Ctrl+C로 중단)")
    print("=" * 80)
    uvicorn.run(main.app, host="0.0.0.0", port=8000, log_level="debug")  # 디버그 로그 활성화
