# ====================================================================
#  File: models/voice.py
# ====================================================================

from typing import Any

import numpy as np
import pyworld
from numpy.typing import NDArray

def adjust_voice(
    fs: int,
    wave: NDArray[Any],
    pitch_scale: float = 1.0,
    intonation_scale: float = 1.0,
) -> tuple[int, NDArray[Any]]:
    """
    주어진 음성 데이터의 피치와 억양을 조정합니다.
    변경하면 음질이 약간 저하될 수 있으므로, 두 값 모두 초기값인 경우에는 그대로 반환합니다.

    Args:
        fs (int): 음성의 샘플링 주파수
        wave (NDArray[Any]): 음성 데이터
        pitch_scale (float, optional): 피치의 높이. (기본값: 1.0)
        intonation_scale (float, optional): 억양의 평균으로부터의 변경 비율. (기본값: 1.0)

    Returns:
        tuple[int, NDArray[Any]]: 조정된 음성 데이터의 샘플링 주파수와 음성 데이터
    """

    if pitch_scale == 1.0 and intonation_scale == 1.0:
        return fs, wave

    wave = wave.astype(np.double)

    f0, t = pyworld.harvest(wave, fs)

    sp = pyworld.cheaptrick(wave, f0, t, fs)
    ap = pyworld.d4c(wave, f0, t, fs)

    non_zero_f0 = [f for f in f0 if f != 0]
    f0_mean = sum(non_zero_f0) / len(non_zero_f0)

    for i, f in enumerate(f0):
        if f == 0:
            continue
        f0[i] = pitch_scale * f0_mean + intonation_scale * (f - f0_mean)

    wave = pyworld.synthesize(f0, sp, ap, fs)
    return fs, wave
