import torch

print(f"PyTorch 버전: {torch.__version__}")
print(f"XPU 사용 가능: {torch.xpu.is_available()}")

if torch.xpu.is_available():
    device_count = torch.xpu.device_count()
    print(f"XPU 개수: {device_count}")
    
    for i in range(device_count):
        device_name = torch.xpu.get_device_name(i)
        print(f"XPU {i}: {device_name if device_name else 'Unknown'}")
        
        # XPU 테스트
        print(f"\n[XPU {i} 테스트]")
        try:
            x = torch.randn(1000, 1000, device=f'xpu:{i}')
            y = torch.randn(1000, 1000, device=f'xpu:{i}')
            z = torch.matmul(x, y)
            print(f"✓ 행렬 곱셈 성공! 결과 shape: {z.shape}")
            print(f"✓ 결과 디바이스: {z.device}")
        except Exception as e:
            print(f"✗ 테스트 실패: {e}")
else:
    print("✗ XPU를 사용할 수 없습니다")
