import torch

print(f"torch.xpu exists: {hasattr(torch, 'xpu')}")

if hasattr(torch, 'xpu'):
    print(f"torch.xpu.is_available: {torch.xpu.is_available()}")
    if torch.xpu.is_available():
        print(f"XPU device count: {torch.xpu.device_count()}")
        if torch.xpu.device_count() > 0:
            print(f"XPU device name: {torch.xpu.get_device_name(0)}")
    else:
        print("XPU is not available")
else:
    print("torch.xpu module not found")
