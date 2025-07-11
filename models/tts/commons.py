# ====================================================================
#  File: models/tts/commons.py
# ====================================================================

from typing import Any, Optional, Union

import torch
from torch.nn import functional as F


def init_weights(m: torch.nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    """
    모듈의 가중치를 정규분포로 초기화합니다.

    Args:
        m (torch.nn.Module): 가중치 초기화 대상 모듈
        mean (float): 정규분포의 평균
        std (float): 정규분포의 표준편차
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """
    커널 크기와 팽창 비율로 패딩 크기를 계산합니다.

    Args:
        kernel_size (int): 커널의 크기
        dilation (int): 팽창 비율

    Returns:
        int: 계산된 패딩의 크기
    """
    return int((kernel_size * dilation - dilation) / 2)

def convert_pad_shape(pad_shape: list[list[Any]]) -> list[Any]:
    """
    패딩의 형태를 변환합니다.

    Args:
        pad_shape (list[list[Any]]): 변환 전 패딩 형태

    Returns:
        list[Any]: 변환 후 패딩의 형태
    """
    layer = pad_shape[::-1]
    new_pad_shape = [item for sublist in layer for item in sublist]
    return new_pad_shape

def intersperse(lst: list[Any], item: Any) -> list[Any]:
    """
    리스트의 요소 사이에 특정 아이템을 삽입합니다.

    Args:
        lst (list[Any]): 원래 리스트
        item (Any): 삽입할 아이템

    Returns:
        list[Any]: 새로운 리스트
    """
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result

def slice_segments(
    x: torch.Tensor, ids_str: torch.Tensor, segment_size: int = 4
) -> torch.Tensor:
    """
    텐서에서 세그먼트를 슬라이스합니다.

    Args:
        x (torch.Tensor): 입력 텐서
        ids_str (torch.Tensor): 슬라이스를 시작할 인덱스
        segment_size (int, optional): 슬라이스의 크기 (기본값: 4)

    Returns:
        torch.Tensor: 슬라이스된 세그먼트
    """
    gather_indices = ids_str.view(x.size(0), 1, 1).repeat(
        1, x.size(1), 1
    ) + torch.arange(segment_size, device=x.device)
    return torch.gather(x, 2, gather_indices)

def rand_slice_segments(
    x: torch.Tensor, x_lengths: Optional[torch.Tensor] = None, segment_size: int = 4
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    랜덤한 세그먼트를 슬라이스합니다.

    Args:
        x (torch.Tensor): 입력 텐서
        x_lengths (Optional[torch.Tensor], optional): 각 배치의 길이 (기본값: None)
        segment_size (int, optional): 슬라이스의 크기 (기본값: 4)

    Returns:
        tuple[torch.Tensor, torch.Tensor]: 슬라이스된 세그먼트와 시작 인덱스
    """
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t 
    ids_str_max = torch.clamp(x_lengths - segment_size + 1, min=0)
    ids_str = (torch.rand([b], device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str

def subsequent_mask(length: int) -> torch.Tensor:
    """
    후속 마스크를 생성합니다.

    Args:
        length (int): 마스크의 크기

    Returns:
        torch.Tensor: 생성된 마스크
    """
    mask = torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)
    return mask

@torch.jit.script
def fused_add_tanh_sigmoid_multiply(
    input_a: torch.Tensor, input_b: torch.Tensor, n_channels: torch.Tensor
) -> torch.Tensor:
    """
    add、tanh、sigmoid의 연산을 결합하여 곱셈을 수행합니다.

    Args:
        input_a (torch.Tensor): 입력 텐서 A
        input_b (torch.Tensor): 입력 텐서 B
        n_channels (torch.Tensor): 채널 수

    Returns:
        torch.Tensor: 연산 결과
    """
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts

def sequence_mask(
    length: torch.Tensor, max_length: Optional[int] = None
) -> torch.Tensor:
    """
    시퀀스 마스크를 생성합니다.

    Args:
        length (torch.Tensor): 각 시퀀스의 길이
        max_length (Optional[int]): 최대의 시퀀스 길이. 지정되지 않은 경우는 length의 최대 값을 사용

    Returns:
        torch.Tensor: 생성된 시퀀스 마스크
    """
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)

def generate_path(duration: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    경로를 생성합니다.

    Args:
        duration (torch.Tensor): 각 시간 단계의 지속 시간
        mask (torch.Tensor): 마스크 텐서

    Returns:
        torch.Tensor: 생성된 경로
    """
    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path.unsqueeze(1).transpose(2, 3) * mask
    return path

def clip_grad_value_(
    parameters: Union[torch.Tensor, list[torch.Tensor]],
    clip_value: Optional[float],
    norm_type: float = 2.0,
) -> float:
    """
    그래디언트 값을 클리핑합니다.

    Args:
        parameters (Union[torch.Tensor, list[torch.Tensor]]): 클리핑할 파라미터
        clip_value (Optional[float]): 클리핑할 값. None인 경우 클리핑하지 않음
        norm_type (float): 노름의 종류 (기본값: 2.0)

    Returns:
        float: 총 노름
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if clip_value is not None:
        clip_value = float(clip_value)

    total_norm = 0.0
    for p in parameters:
        assert p.grad is not None
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
        if clip_value is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm
