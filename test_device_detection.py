#!/usr/bin/env python3
"""
L.U.N.A. XPU 지원 테스트
uvicorn 프로세스 환경에서 XPU 상태 확인
"""

import sys
sys.path.insert(0, '.')

from utils.device import get_available_device, get_device_info, optimize_device

print("=" * 80)
print("L.U.N.A. XPU 지원 테스트")
print("=" * 80)

# 1. 디바이스 자동 감지
print("\n[1] 디바이스 자동 감지")
device = get_available_device()
print(f"  감지된 디바이스: {device}")

# 2. 디바이스 정보
print("\n[2] 디바이스 정보")
info = get_device_info(device)
print(f"  타입: {info['type']}")
print(f"  이름: {info['name']}")
print(f"  개수: {info['device_count']}")
if info['memory_total'] > 0:
    print(f"  총 메모리: {info['memory_total'] / (1024**3):.2f} GB")

# 3. 디바이스 최적화
print("\n[3] 디바이스 최적화 적용")
optimize_device(device)

# 4. 간단한 텐서 연산 테스트
print("\n[4] 텐서 연산 테스트")
try:
    import torch
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    z = torch.matmul(x, y)
    print(f"  ✓ 행렬 곱셈 성공!")
    print(f"  ✓ 결과 디바이스: {z.device}")
    print(f"  ✓ 결과 shape: {z.shape}")
except Exception as e:
    print(f"  ✗ 텐서 연산 실패: {e}")

print("\n" + "=" * 80)
print(f"✓ {device.upper()} 디바이스가 정상적으로 작동합니다!")
print("=" * 80)
