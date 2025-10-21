# ====================================================================
#  File: models/tts/utils/__init__.py
# --------------------------------------------------------------------
#  TTS 모델의 유틸리티를 정의하는 모듈입니다.
#
#  NOTE: 이 코드는 해당 프로젝트에 특화된 기능을 포함하고 있으며,
#        다른 프로젝트에서는 동작하지 않을 수 있습니다.
# ====================================================================

import glob
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

from numpy.typing import NDArray

from models.tts.utils import checkpoints  # type: ignore
from models.tts.utils import safetensors  # type: ignore

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

__is_matplotlib_imported = False

def summarize(
    writer: "SummaryWriter",
    global_step: int,
    scalars: dict[str, float] = {},
    histograms: dict[str, Any] = {},
    images: dict[str, Any] = {},
    audios: dict[str, Any] = {},
    audio_sampling_rate: int = 22050,
) -> None:
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)

def is_resuming(dir_path: Union[str, Path]) -> bool:
    g_list = glob.glob(os.path.join(dir_path, "G_*.pth"))
    return len(g_list) > 0

def plot_spectrogram_to_numpy(spectrogram: NDArray[Any]) -> NDArray[Any]:
    global __is_matplotlib_imported
    if not __is_matplotlib_imported:
        import matplotlib

        matplotlib.use("Agg")
        __is_matplotlib_imported = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")  # type: ignore
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data

def plot_alignment_to_numpy(
    alignment: NDArray[Any], info: Optional[str] = None
) -> NDArray[Any]:
    global __is_matplotlib_imported
    if not __is_matplotlib_imported:
        import matplotlib

        matplotlib.use("Agg")
        __is_matplotlib_imported = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(
        alignment.transpose(), aspect="auto", origin="lower", interpolation="none"
    )
    fig.colorbar(im, ax=ax)
    xlabel = "Decoder timestep"
    if info is not None:
        xlabel += "\n\n" + info
    plt.xlabel(xlabel)
    plt.ylabel("Encoder timestep")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")  # type: ignore
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data

def load_wav_to_torch(full_path: Union[str, Path]) -> tuple[torch.FloatTensor, int]:
    try:
        from scipy.io.wavfile import read
    except ImportError:
        raise ImportError("[L.U.N.A.] wav 파일을 불러오기 위해 scipy가 필요합니다. pip install scipy 로 설치해주세요.")

    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate

def load_filepaths_and_text(
    filename: Union[str, Path], split: str = "|"
) -> list[list[str]]:
    with open(filename, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text

def get_logger(
    model_dir_path: Union[str, Path], filename: str = "train.log"
) -> logging.Logger:
    global logger
    logger = logging.getLogger(os.path.basename(model_dir_path))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)
    h = logging.FileHandler(os.path.join(model_dir_path, filename))
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)
    return logger

def get_steps(model_path: Union[str, Path]) -> Optional[int]:
    matches = re.findall(r"\d+", model_path)  # type: ignore
    return matches[-1] if matches else None

def check_git_hash(model_dir_path: Union[str, Path]) -> None:
    source_dir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(os.path.join(source_dir, ".git")):
        logger.warning(
            f"[L.U.N.A.] {source_dir}는 Git 저장소가 아닙니다. Git Hash 체크를 건너뜁니다."
        )
        return

    cur_hash = subprocess.getoutput("git rev-parse HEAD")

    path = os.path.join(model_dir_path, "githash")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            saved_hash = f.read()
        if saved_hash != cur_hash:
            logger.warning(
                f"[L.U.N.A.] Git Hash 값이 다릅니다. {saved_hash[:8]}(saved) != {cur_hash[:8]}(current)"
            )
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(cur_hash)
