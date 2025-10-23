#!/usr/bin/env python3
"""
L.U.N.A. 디바이스 확인 스크립트

현재 시스템에서 사용 가능한 AI 가속기를 확인합니다.
- Intel XPU (Arc GPU)
- NVIDIA CUDA
- AMD ROCm
- Apple MPS
- CPU
"""

import sys
import platform

print("=" * 80)
print("L.U.N.A. 시스템 디바이스 확인")
print("=" * 80)

# 1. 시스템 정보
print("\n[1] 시스템 정보")
print(f"  OS: {platform.system()} {platform.release()}")
print(f"  Python: {sys.version.split()[0]}")
print(f"  Architecture: {platform.machine()}")

# 2. PyTorch 확인
print("\n[2] PyTorch 확인")
try:
    import torch
    print(f"  ✓ PyTorch 버전: {torch.__version__}")
    print(f"  ✓ PyTorch 빌드 정보: {torch.version.debug}")
except ImportError as e:
    print(f"  ✗ PyTorch가 설치되지 않았습니다: {e}")
    sys.exit(1)

# 3. Intel XPU (Arc GPU) 확인
print("\n[3] Intel XPU (Arc GPU) 확인")
try:
    import intel_extension_for_pytorch as ipex
    print(f"  ✓ IPEX 버전: {ipex.__version__}")
    
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        print(f"  ✓ Intel XPU 사용 가능!")
        device_count = torch.xpu.device_count()
        print(f"  ✓ 감지된 XPU 개수: {device_count}")
        
        for i in range(device_count):
            try:
                device_name = torch.xpu.get_device_name(i)
                if device_name:
                    print(f"    - XPU {i}: {device_name}")
                else:
                    print(f"    - XPU {i}: (이름 조회 불가)")
                
                # 메모리 정보
                try:
                    props = torch.xpu.get_device_properties(i)
                    if hasattr(props, 'total_memory'):
                        total_memory = props.total_memory / (1024**3)
                        print(f"      총 메모리: {total_memory:.2f} GB")
                except:
                    print(f"      메모리 정보: 조회 불가")
            except Exception as e:
                print(f"    - XPU {i}: 정보 조회 실패 ({e})")
        
        # XPU 테스트
        try:
            print("\n  [XPU 간단 테스트]")
            x = torch.randn(100, 100, device='xpu')
            y = torch.randn(100, 100, device='xpu')
            z = torch.matmul(x, y)
            print(f"    ✓ 행렬 곱셈 테스트 성공 (결과 shape: {z.shape})")
            print(f"    ✓ 결과 디바이스: {z.device}")
        except Exception as e:
            print(f"    ✗ XPU 테스트 실패: {e}")
    else:
        print("  ✗ Intel XPU를 사용할 수 없습니다")
        print("    - IPEX는 설치되어 있지만 XPU 디바이스가 감지되지 않습니다")
except ImportError:
    print("  ✗ Intel Extension for PyTorch (IPEX)가 설치되지 않았습니다")
    print("    설치 방법: pip install intel-extension-for-pytorch")
except Exception as e:
    print(f"  ✗ XPU 확인 중 오류 발생: {e}")

# 4. NVIDIA CUDA 확인
print("\n[4] NVIDIA CUDA 확인")
if torch.cuda.is_available():
    print(f"  ✓ CUDA 사용 가능!")
    print(f"  ✓ CUDA 버전: {torch.version.cuda}")
    print(f"  ✓ cuDNN 버전: {torch.backends.cudnn.version()}")
    device_count = torch.cuda.device_count()
    print(f"  ✓ 감지된 GPU 개수: {device_count}")
    
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory / (1024**3)
        print(f"    - GPU {i}: {device_name}")
        print(f"      총 메모리: {total_memory:.2f} GB")
else:
    print("  ✗ CUDA를 사용할 수 없습니다")

# 5. AMD ROCm 확인
print("\n[5] AMD ROCm 확인")
try:
    if hasattr(torch, 'hip') and torch.hip.is_available():
        print(f"  ✓ ROCm 사용 가능!")
        print(f"  ✓ ROCm 버전: {torch.version.hip}")
    else:
        print("  ✗ ROCm을 사용할 수 없습니다")
except:
    print("  ✗ ROCm을 사용할 수 없습니다")

# 6. Apple MPS 확인
print("\n[6] Apple MPS (Metal) 확인")
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print("  ✓ MPS 사용 가능!")
    if torch.backends.mps.is_built():
        print("  ✓ MPS 백엔드 빌드됨")
else:
    print("  ✗ MPS를 사용할 수 없습니다 (macOS 전용)")

# 7. CPU 정보
print("\n[7] CPU 정보")
try:
    import psutil
    cpu_count = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq()
    ram = psutil.virtual_memory()
    print(f"  ✓ CPU 코어 수: {cpu_count}")
    if cpu_freq:
        print(f"  ✓ CPU 주파수: {cpu_freq.current:.2f} MHz (최대: {cpu_freq.max:.2f} MHz)")
    print(f"  ✓ RAM: {ram.total / (1024**3):.2f} GB (사용 가능: {ram.available / (1024**3):.2f} GB)")
except ImportError:
    print("  ! psutil이 설치되지 않아 상세 정보를 표시할 수 없습니다")
    print("    설치 방법: pip install psutil")

# 8. 권장 디바이스
print("\n" + "=" * 80)
print("권장 설정")
print("=" * 80)

recommended_device = "cpu"
if hasattr(torch, 'xpu') and torch.xpu.is_available():
    recommended_device = "xpu"
    print("✓ Intel XPU (Arc GPU)를 사용하는 것을 권장합니다")
    print("  config/models.yaml에서:")
    print('    tts:')
    print('      device: "xpu"')
elif torch.cuda.is_available():
    recommended_device = "cuda"
    print("✓ NVIDIA CUDA GPU를 사용하는 것을 권장합니다")
    print("  config/models.yaml에서:")
    print('    tts:')
    print('      device: "cuda"')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    recommended_device = "mps"
    print("✓ Apple MPS (Metal)를 사용하는 것을 권장합니다")
    print("  config/models.yaml에서:")
    print('    tts:')
    print('      device: "mps"')
else:
    print("! 가속기가 감지되지 않았습니다. CPU를 사용합니다")
    print("  성능이 느릴 수 있습니다")

print(f"\n현재 권장 디바이스: {recommended_device}")
print("=" * 80)
