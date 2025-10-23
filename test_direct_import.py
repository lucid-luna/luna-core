"""main.py를 직접 import해서 XPU 상태 확인"""
import torch

print("=== 1. Before any imports ===")
print(f"XPU device count: {torch.xpu.device_count()}")

print("\n=== 2. Importing main module ===")
import main

print("\n=== 3. After importing main ===")
print(f"XPU device count: {torch.xpu.device_count()}")
