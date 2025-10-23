"""서버 import 컨텍스트에서 XPU 상태 확인"""
import torch

print("=== Before imports ===")
print(f"XPU available: {torch.xpu.is_available()}")
print(f"XPU device count: {torch.xpu.device_count()}")

print("\n=== Importing utils.device ===")
from utils.device import get_available_device, get_device_info

device = get_available_device()
print(f"Detected device: {device}")
print(f"Device info: {get_device_info(device)}")

print("\n=== After utils.device ===")
print(f"XPU available: {torch.xpu.is_available()}")
print(f"XPU device count: {torch.xpu.device_count()}")

print("\n=== Importing services.tts ===")
try:
    from services.tts import TTSService
    print("✓ TTS import successful")
except Exception as e:
    print(f"✗ TTS import failed: {e}")

print(f"XPU available: {torch.xpu.is_available()}")
print(f"XPU device count: {torch.xpu.device_count()}")
