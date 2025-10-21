# ====================================================================
#  File: models/tts/utils/checkpoints.py
# --------------------------------------------------------------------
#  TTS 모델의 체크포인트 유틸리티를 정의하는 모듈입니다.
#
#  NOTE: 이 코드는 해당 프로젝트에 특화된 기능을 포함하고 있으며,
#        다른 프로젝트에서는 동작하지 않을 수 있습니다.
# ====================================================================

import glob
import os
import re
import logging
from pathlib import Path
from typing import Any, Optional, Union

import torch

logger = logging.getLogger(__name__)

def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    skip_optimizer: bool = False,
    for_infer: bool = False,
) -> tuple[torch.nn.Module, Optional[torch.optim.Optimizer], float, int]:
    """
    경로에서 체크포인트를 로드하고 모델과 옵티마이저를 업데이트합니다.

    Args:
        checkpoint_path (Union[str, Path]): 체크포인트 파일의 경로
        model (torch.nn.Module): 업데이트할 모델
        optimizer (Optional[torch.optim.Optimizer]): 업데이트할 옵티마이저. None인 경우 업데이트하지 않음
        skip_optimizer (bool): 옵티마이저의 업데이트를 스킵할지 여부
        for_infer (bool): 추론용으로 로드할지 여부

    Returns:
        tuple[torch.nn.Module, Optional[torch.optim.Optimizer], float, int]: 업데이트된 모델과 옵티마이저, 학습률, 이터레이션 횟수
    """

    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]
    logger.info(
        f"Loading model and optimizer at iteration {iteration} from {checkpoint_path}"
    )
    if (
        optimizer is not None
        and not skip_optimizer
        and checkpoint_dict["optimizer"] is not None
    ):
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    elif optimizer is None and not skip_optimizer:
        # else:      Disable this line if Infer and resume checkpoint,then enable the line upper
        new_opt_dict = optimizer.state_dict()  # type: ignore
        new_opt_dict_params = new_opt_dict["param_groups"][0]["params"]
        new_opt_dict["param_groups"] = checkpoint_dict["optimizer"]["param_groups"]
        new_opt_dict["param_groups"][0]["params"] = new_opt_dict_params
        optimizer.load_state_dict(new_opt_dict)  # type: ignore

    saved_state_dict = checkpoint_dict["model"]
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            # assert "emb_g" not in k
            new_state_dict[k] = saved_state_dict[k]
            assert saved_state_dict[k].shape == v.shape, (
                saved_state_dict[k].shape,
                v.shape,
            )
        except:
            # For upgrading from the old version
            if "ja_bert_proj" in k:
                v = torch.zeros_like(v)
                logger.warning(
                    f"Seems you are using the old version of the model, the {k} is automatically set to zero for backward compatibility"
                )
            elif "enc_q" in k and for_infer:
                continue
            else:
                logger.error(f"{k} is not in the checkpoint {checkpoint_path}")

            new_state_dict[k] = v

    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(new_state_dict, strict=False)

    logger.info(f"Loaded '{checkpoint_path}' (iteration {iteration})")

    return model, optimizer, learning_rate, iteration


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Union[torch.optim.Optimizer, torch.optim.AdamW],
    learning_rate: float,
    iteration: int,
    checkpoint_path: Union[str, Path],
) -> None:
    """
    모델과 옵티마이저의 상태를 지정된 경로에 저장합니다.

    Args:
        model (torch.nn.Module): 저장할 모델
        optimizer (Union[torch.optim.Optimizer, torch.optim.AdamW]): 저장할 옵티마이저
        learning_rate (float): 학습률
        iteration (int): 이터레이션 횟수
        checkpoint_path (Union[str, Path]): 저장할 경로
    """
    logger.info(
        f"Saving model and optimizer state at iteration {iteration} to {checkpoint_path}"
    )
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(
        {
            "model": state_dict,
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        checkpoint_path,
    )


def clean_checkpoints(
    model_dir_path: Union[str, Path] = "logs/44k/",
    n_ckpts_to_keep: int = 2,
    sort_by_time: bool = True,
) -> None:
    """
    체크포인트 디렉토리에서 오래된 체크포인트 파일을 삭제하여 디스크 공간을 확보합니다.

    Args:
        model_dir_path (Union[str, Path]): 모델이 저장된 디렉토리의 경로
        n_ckpts_to_keep (int): 유지할 체크포인트의 수 (G_0.pth 및 D_0.pth 제외)
        sort_by_time (bool): True인 경우 시간순으로 삭제, False인 경우 이름순으로 삭제
    """

    ckpts_files = [
        f
        for f in os.listdir(model_dir_path)
        if os.path.isfile(os.path.join(model_dir_path, f))
    ]

    def name_key(_f: str) -> int:
        return int(re.compile("._(\\d+)\\.pth").match(_f).group(1))  # type: ignore

    def time_key(_f: str) -> float:
        return os.path.getmtime(os.path.join(model_dir_path, _f))

    sort_key = time_key if sort_by_time else name_key

    def x_sorted(_x: str) -> list[str]:
        return sorted(
            [f for f in ckpts_files if f.startswith(_x) and not f.endswith("_0.pth")],
            key=sort_key,
        )

    to_del = [
        os.path.join(model_dir_path, fn)
        for fn in (
            x_sorted("G_")[:-n_ckpts_to_keep]
            + x_sorted("D_")[:-n_ckpts_to_keep]
            + x_sorted("WD_")[:-n_ckpts_to_keep]
            + x_sorted("DUR_")[:-n_ckpts_to_keep]
        )
    ]

    def del_info(fn: str) -> None:
        return logger.info(f"Free up space by deleting ckpt {fn}")

    def del_routine(x: str) -> list[Any]:
        return [os.remove(x), del_info(x)]

    [del_routine(fn) for fn in to_del]


def get_latest_checkpoint_path(
    model_dir_path: Union[str, Path], regex: str = "G_*.pth"
) -> str:
    """
    지정된 디렉토리에서 최신의 체크포인트의 경로를 가져옵니다.

    Args:
        model_dir_path (Union[str, Path]): 모델이 저장된 디렉토리의 경로
        regex (str): 체크포인트의 파일 이름에 대한 정규 표현식

    Returns:
        str: 최신의 체크포인트의 경로
    """

    f_list = glob.glob(os.path.join(str(model_dir_path), regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    try:
        x = f_list[-1]
    except IndexError:
        raise ValueError(f"No checkpoint found in {model_dir_path} with regex {regex}")

    return x
