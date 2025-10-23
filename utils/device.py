# ====================================================================
#  File: utils/device.py
# ====================================================================
"""
디바이스 감지 및 관리 유틸리티

Intel XPU, CUDA, CPU 자동 감지 및 최적화
"""

import logging
import torch
from typing import Optional

logger = logging.getLogger(__name__)

def get_available_device(preferred_device: Optional[str] = None) -> str:
    """
    사용 가능한 최적의 디바이스를 반환합니다.
    
    우선순위:
    1. Intel XPU (Arc GPU)
    2. CUDA (NVIDIA GPU)
    3. CPU
    
    Args:
        preferred_device: 선호하는 디바이스 ("xpu", "cuda", "cpu")
    
    Returns:
        str: 사용 가능한 디바이스 ("xpu", "cuda", "cpu")
    """
    # 선호 디바이스가 지정된 경우
    if preferred_device:
        if preferred_device == "xpu":
            try:
                if hasattr(torch, 'xpu') and torch.xpu.is_available():
                    device_count = torch.xpu.device_count()
                    logger.info(f"Intel XPU 감지됨 ({device_count}개 디바이스)")
                    return "xpu"
                else:
                    logger.warning("Intel XPU가 요청되었지만 사용 불가능합니다. CPU로 대체합니다.")
            except Exception as e:
                logger.warning(f"Intel XPU 확인 실패: {e}. CPU로 대체합니다.")
        elif preferred_device == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            else:
                logger.warning("CUDA가 요청되었지만 사용 불가능합니다. CPU로 대체합니다.")
        return "cpu"
    
    # 자동 감지
    # 1. Intel XPU 확인 (IPEX 없이도 PyTorch XPU 지원 확인)
    try:
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            # device_count()가 0을 리턴하는 버그 우회: is_available()만 신뢰
            try:
                device_count = torch.xpu.device_count()
                if device_count > 0:
                    logger.info(f"✓ Intel XPU 감지됨 ({device_count}개 디바이스)")
                else:
                    # device_count가 0이어도 is_available()이 True면 XPU 사용 가능
                    logger.warning(f"⚠ XPU device_count=0이지만 is_available=True, XPU 사용 시도")
                    device_count = 1  # Fallback
            except Exception as e:
                logger.warning(f"XPU device_count 확인 실패: {e}, is_available() 신뢰")
                device_count = 1
            
            # IPEX가 있으면 추가 최적화 가능
            try:
                import intel_extension_for_pytorch as ipex
                logger.debug("intel_extension_for_pytorch 로드됨")
            except:
                logger.debug("intel_extension_for_pytorch 없이 XPU 사용")
            return "xpu"
    except Exception as e:
        logger.debug(f"Intel XPU 확인 중 오류: {e}")
    
    # 2. CUDA 확인
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info(f"✓ CUDA 감지됨 ({device_count}개 디바이스)")
        return "cuda"
    
    # 3. CPU 사용
    logger.info("CPU 모드로 실행")
    return "cpu"


def optimize_device(device: str):
    """
    디바이스별 최적화를 적용합니다.
    
    Args:
        device: 디바이스 타입 ("xpu", "cuda", "cpu")
    """
    if device == "xpu":
        # Intel XPU 최적화 설정
        logger.info("Intel XPU 최적화 적용")
        # IPEX가 있으면 추가 최적화 적용
        try:
            import intel_extension_for_pytorch as ipex
            logger.debug("IPEX 최적화 사용 가능")
            # ipex.optimize() 는 모델별로 적용
        except:
            logger.debug("IPEX 없이 기본 XPU 사용")
    
    elif device == "cuda":
        # CUDA 최적화 설정
        if torch.cuda.is_available():
            # CuDNN 자동 튜닝
            torch.backends.cudnn.benchmark = True
            logger.info("CUDA 최적화 적용 (cuDNN benchmark 활성화)")
    
    elif device == "cpu":
        # CPU 최적화 설정
        torch.set_num_threads(torch.get_num_threads())
        logger.info(f"CPU 최적화 적용 (스레드: {torch.get_num_threads()})")


def to_device(tensor_or_model, device: str):
    """
    텐서 또는 모델을 지정된 디바이스로 이동합니다.
    
    Args:
        tensor_or_model: 이동할 텐서 또는 모델
        device: 목표 디바이스 ("xpu", "cuda", "cpu")
    
    Returns:
        디바이스로 이동된 객체
    """
    if device == "xpu":
        try:
            if hasattr(tensor_or_model, 'to'):
                return tensor_or_model.to("xpu")
            return tensor_or_model
        except Exception as e:
            logger.warning(f"Intel XPU로 이동 실패: {e}, CPU 사용")
            return tensor_or_model.to("cpu") if hasattr(tensor_or_model, 'to') else tensor_or_model
    else:
        return tensor_or_model.to(device) if hasattr(tensor_or_model, 'to') else tensor_or_model


def get_device_info(device: str) -> dict:
    """
    디바이스 정보를 반환합니다.
    
    Args:
        device: 디바이스 타입
    
    Returns:
        dict: 디바이스 정보
    """
    info = {
        "type": device,
        "name": "Unknown",
        "memory_total": 0,
        "memory_allocated": 0,
        "device_count": 0
    }
    
    try:
        if device == "xpu":
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                info["device_count"] = torch.xpu.device_count()
                try:
                    device_name = torch.xpu.get_device_name(0)
                    info["name"] = device_name if device_name else f"Intel Arc GPU x{info['device_count']}"
                except:
                    info["name"] = f"Intel Arc GPU x{info['device_count']}"
                # XPU 메모리 정보 (API 지원 시)
        
        elif device == "cuda":
            if torch.cuda.is_available():
                info["device_count"] = torch.cuda.device_count()
                info["name"] = torch.cuda.get_device_name(0)
                info["memory_total"] = torch.cuda.get_device_properties(0).total_memory
                info["memory_allocated"] = torch.cuda.memory_allocated(0)
        
        elif device == "cpu":
            import psutil
            info["name"] = "CPU"
            info["memory_total"] = psutil.virtual_memory().total
            info["memory_allocated"] = psutil.virtual_memory().used
    
    except Exception as e:
        logger.warning(f"디바이스 정보 수집 실패: {e}")
    
    return info


def empty_cache(device: str):
    """
    디바이스 캐시를 비웁니다.
    
    Args:
        device: 디바이스 타입
    """
    try:
        if device == "xpu":
            if hasattr(torch, 'xpu') and hasattr(torch.xpu, 'empty_cache'):
                torch.xpu.empty_cache()
        elif device == "cuda":
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except Exception as e:
        logger.debug(f"캐시 비우기 실패: {e}")
