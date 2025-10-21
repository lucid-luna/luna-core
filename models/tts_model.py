# ====================================================================
#  File: models/tts_model.py
# ====================================================================

import logging
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel

from models.nlp.constants import (
    DEFAULT_ASSIST_TEXT_WEIGHT,
    DEFAULT_LENGTH,
    DEFAULT_LINE_SPLIT,
    DEFAULT_NOISE,
    DEFAULT_NOISEW,
    DEFAULT_SDP_RATIO,
    DEFAULT_SPLIT_INTERVAL,
    DEFAULT_STYLE,
    DEFAULT_STYLE_WEIGHT,
    Languages,
)

from models.tts.hyper_parameters import HyperParameters
from models.tts.infer import get_net_g, infer
from models.tts.models import SynthesizerTrn
from models.tts.models_jp_extra import (
    SynthesizerTrn as SynthesizerTrnJPExtra,
)
from models.voice import adjust_voice

logger = logging.getLogger(__name__)

class TTSModel:
    """
    LunaTTS 모델의 음성 합성 기능을 제공하는 클래스입니다.
    모델/하이퍼파라미터/스타일 벡터의 경로와 장치를 지정하여 초기화하며, model.infer() 메서드를 호출하여 음성 합성을 수행할 수 있습니다.
    """

    def __init__(
        self,
        model_path: Path,
        config_path: Union[Path, HyperParameters],
        style_vec_path: Union[Path, NDArray[Any]],
        device: str,
    ) -> None:
        """
        LunaTTS 모델을 초기화합니다.

        Args:
            model_path (Path): 모델 (.safetensors) 의 경로
            config_path (Union[Path, HyperParameters]): 하이퍼파라미터 (config.json) 의 경로 (직접 HyperParameters 를 지정하는 것도 가능)
            style_vec_path (Union[Path, NDArray[Any]]): 스타일 벡터 (style_vectors.npy) 의 경로 (직접 NDArray 를 지정하는 것도 가능)
            device (str): 음성 합성 시에 사용할 장치 (cpu, cuda, mps 등)
        """

        self.model_path: Path = model_path
        self.device: str = device

        if isinstance(config_path, HyperParameters):
            self.config_path: Path = Path("")
            self.hyper_parameters: HyperParameters = config_path
        else:
            self.config_path: Path = config_path
            self.hyper_parameters: HyperParameters = HyperParameters.load_from_json(
                self.config_path
            )

        if isinstance(style_vec_path, np.ndarray):
            self.style_vec_path: Path = Path("")
            self.__style_vectors: NDArray[Any] = style_vec_path
        else:
            self.style_vec_path: Path = style_vec_path
            self.__style_vectors: NDArray[Any] = np.load(self.style_vec_path)

        self.spk2id: dict[str, int] = self.hyper_parameters.data.spk2id
        self.id2spk: dict[int, str] = {v: k for k, v in self.spk2id.items()}

        num_styles: int = self.hyper_parameters.data.num_styles
        if hasattr(self.hyper_parameters.data, "style2id"):
            self.style2id: dict[str, int] = self.hyper_parameters.data.style2id
        else:
            self.style2id: dict[str, int] = {str(i): i for i in range(num_styles)}
        if len(self.style2id) != num_styles:
            raise ValueError(
                f"스타일 수 ({num_styles})가 style2id의 수 ({len(self.style2id)})와 일치하지 않습니다."
            )

        if self.__style_vectors.shape[0] != num_styles:
            raise ValueError(
                f"스타일 수 ({num_styles})가 스타일 벡터 수 ({self.__style_vectors.shape[0]})와 일치하지 않습니다."
            )
        self.__style_vector_inference: Optional[Any] = None

        self.__net_g: Union[SynthesizerTrn, SynthesizerTrnJPExtra, None] = None

    def load(self) -> None:
        self.__net_g = get_net_g(
            model_path=str(self.model_path),
            version=self.hyper_parameters.version,
            device=self.device,
            hps=self.hyper_parameters,
        )

    def __get_style_vector(self, style_id: int, weight: float = 1.0) -> NDArray[Any]:
        mean = self.__style_vectors[0]
        style_vec = self.__style_vectors[style_id]
        style_vec = mean + (style_vec - mean) * weight
        return style_vec

    def __get_style_vector_from_audio(
        self, audio_path: str, weight: float = 1.0
    ) -> NDArray[Any]:
        if self.__style_vector_inference is None:
            try:
                import pyannote.audio
            except ImportError:
                raise ImportError(
                    "pyannote.audio는 스타일 벡터를 오디오에서 추론하는 데 필요합니다."
                )

            self.__style_vector_inference = pyannote.audio.Inference(
                model=pyannote.audio.Model.from_pretrained(
                    "pyannote/wespeaker-voxceleb-resnet34-LM"
                ),
                window="whole",
            )
            self.__style_vector_inference.to(torch.device(self.device))

        xvec = self.__style_vector_inference(audio_path)
        mean = self.__style_vectors[0]
        xvec = mean + (xvec - mean) * weight
        return xvec

    def __convert_to_16_bit_wav(self, data: NDArray[Any]) -> NDArray[Any]:
        """
        음성 데이터를 16-bit 정수로 변환합니다.

        Args:
            data (NDArray[Any]): 음성 데이터

        Returns:
            NDArray[Any]: 16-bit 정수 음성 데이터
        """
        if data.dtype in [np.float64, np.float32, np.float16]:  # type: ignore
            data = data / np.abs(data).max()
            data = data * 32767
            data = data.astype(np.int16)
        elif data.dtype == np.int32:
            data = data / 65536
            data = data.astype(np.int16)
        elif data.dtype == np.int16:
            pass
        elif data.dtype == np.uint16:
            data = data - 32768
            data = data.astype(np.int16)
        elif data.dtype == np.uint8:
            data = data * 257 - 32768
            data = data.astype(np.int16)
        elif data.dtype == np.int8:
            data = data * 256
            data = data.astype(np.int16)
        else:
            raise ValueError(
                "오디오 데이터를 "
                f"{data.dtype}에서 16-bit int 형식으로 변환할 수 없습니다."
            )
        return data

    def infer(
        self,
        text: str,
        language: Languages = Languages.JP,
        speaker_id: int = 0,
        reference_audio_path: Optional[str] = None,
        sdp_ratio: float = DEFAULT_SDP_RATIO,
        noise: float = DEFAULT_NOISE,
        noise_w: float = DEFAULT_NOISEW,
        length: float = DEFAULT_LENGTH,
        line_split: bool = DEFAULT_LINE_SPLIT,
        split_interval: float = DEFAULT_SPLIT_INTERVAL,
        assist_text: Optional[str] = None,
        assist_text_weight: float = DEFAULT_ASSIST_TEXT_WEIGHT,
        use_assist_text: bool = False,
        style: str = DEFAULT_STYLE,
        style_weight: float = DEFAULT_STYLE_WEIGHT,
        given_phone: Optional[list[str]] = None,
        given_tone: Optional[list[int]] = None,
        pitch_scale: float = 1.0,
        intonation_scale: float = 1.0,
    ) -> tuple[int, NDArray[Any]]:
        """
        텍스트에서 음성을 생성합니다.

        Args:
            text (str): 읽어야 할 텍스트
            language (Languages, optional): 언어 (기본값: Languages.JP)
            speaker_id (int, optional): 화자 ID. (기본값: 0)
            reference_audio_path (Optional[str], optional): 음성 스타일의 참조 음성 파일 경로 (기본값: None)
            sdp_ratio (float, optional): DP와 SDP의 혼합 비율. 0에서 DP만, 1에서 SDP만 사용 (값이 커질수록 템포에 완급이 생김) (기본값: DEFAULT_SDP_RATIO)
            noise (float, optional): DP에 주어지는 노이즈 (기본값: DEFAULT_NOISE)
            noise_w (float, optional): SDP에 주어지는 노이즈 (기본값: DEFAULT_NOISEW)
            length (float, optional): 생성 음성의 길이(말 속도) 매개변수 클수록 생성 음성이 길고 느려지며, 작을수록 짧고 빨라짐. (기본값: DEFAULT_LENGTH)
            line_split (bool, optional): 텍스트를 개행마다 분할하여 생성할지 여부 (True인 경우 given_phone/given_tone은 무시됨) (기본값: DEFAULT_LINE_SPLIT)
            split_interval (float, optional): 개행마다 분할할 경우의 무음 (초) (기본값: DEFAULT_SPLIT_INTERVAL)
            assist_text (Optional[str], optional): 감정 표현의 참조 원본 보조 텍스트 (기본값: None)
            assist_text_weight (float, optional): 감정 표현의 보조 텍스트를 적용하는 강도 (기본값: DEFAULT_ASSIST_TEXT_WEIGHT)
            use_assist_text (bool, optional): 음성 합성 시 감정 표현의 보조 텍스트를 사용할지 여부 (기본값: False)
            style (str, optional): 음성 스타일 (Neutral, Happy 등) (기본값: DEFAULT_STYLE)
            style_weight (float, optional): 음성 스타일을 적용하는 강도 (기본값: DEFAULT_STYLE_WEIGHT)
            given_phone (Optional[list[int]], optional): 읽어야 할 텍스트의 음소를 나타내는 리스트 지정하는 경우 given_tone도 별도로 지정해야 함. (기본값: None)
            given_tone (Optional[list[int]], optional): 억양의 톤 리스트 (기본값: None)
            pitch_scale (float, optional): 피치의 높이 (1.0에서 변경하면 약간 음질이 저하됨) (기본값: 1.0)
            intonation_scale (float, optional): 억양의 평균에서 변화하는 폭 (1.0에서 변경하면 약간 음질이 저하됨) (기본값: 1.0)

        Returns:
            tuple[int, NDArray[Any]]: 샘플링 레이트와 음성 데이터 (16bit PCM)
        """

        logger.info(f"텍스트에서 오디오 데이터 생성을 시작합니다:\n{text}")
        if language != "JP" and self.hyper_parameters.version.endswith("JP-Extra"):
            raise ValueError(
                "Style-BERT-VITS2 JP-Extra로 훈련된 모델은 일본어 텍스트에 대해서만 사용할 수 있습니다."
            )
        if reference_audio_path == "":
            reference_audio_path = None
        if assist_text == "" or not use_assist_text:
            assist_text = None

        if self.__net_g is None:
            self.load()
        assert self.__net_g is not None
        if reference_audio_path is None:
            style_id = self.style2id[style]
            style_vector = self.__get_style_vector(style_id, style_weight)
        else:
            style_vector = self.__get_style_vector_from_audio(
                reference_audio_path, style_weight
            )
        if not line_split:
            with torch.no_grad():
                audio = infer(
                    text=text,
                    sdp_ratio=sdp_ratio,
                    noise_scale=noise,
                    noise_scale_w=noise_w,
                    length_scale=length,
                    sid=speaker_id,
                    language=language,
                    hps=self.hyper_parameters,
                    net_g=self.__net_g,
                    device=self.device,
                    assist_text=assist_text,
                    assist_text_weight=assist_text_weight,
                    style_vec=style_vector,
                    given_phone=given_phone,
                    given_tone=given_tone,
                )
        else:
            texts = text.split("\n")
            texts = [t for t in texts if t != ""]
            audios = []
            with torch.no_grad():
                for i, t in enumerate(texts):
                    audios.append(
                        infer(
                            text=t,
                            sdp_ratio=sdp_ratio,
                            noise_scale=noise,
                            noise_scale_w=noise_w,
                            length_scale=length,
                            sid=speaker_id,
                            language=language,
                            hps=self.hyper_parameters,
                            net_g=self.__net_g,
                            device=self.device,
                            assist_text=assist_text,
                            assist_text_weight=assist_text_weight,
                            style_vec=style_vector,
                        )
                    )
                    if i != len(texts) - 1:
                        audios.append(np.zeros(int(44100 * split_interval)))
                audio = np.concatenate(audios)
        logger.info("오디오 데이터를 성공적으로 생성했습니다.")
        if not (pitch_scale == 1.0 and intonation_scale == 1.0):
            _, audio = adjust_voice(
                fs=self.hyper_parameters.data.sampling_rate,
                wave=audio,
                pitch_scale=pitch_scale,
                intonation_scale=intonation_scale,
            )
        audio = self.__convert_to_16_bit_wav(audio)
        return (self.hyper_parameters.data.sampling_rate, audio)

class TTSModelInfo(BaseModel):
    name: str
    files: list[str]
    styles: list[str]
    speakers: list[str]

class TTSModelHolder:
    """
    LunaTTS 모델을 관리하는 클래스입니다.
    """

    def __init__(self, model_root_dir: Path, device: str) -> None:
        from pathlib import Path as _Path
        self.root_dir: _Path = _Path(model_root_dir)
        
        self.device: str = device
        self.model_files_dict: dict[str, list[Path]] = {}
        self.current_model: Optional[TTSModel] = None
        self.model_names: list[str] = []
        self.models_info: list[TTSModelInfo] = []
        self.refresh()

    def refresh(self) -> None:
        self.model_files_dict = {}
        self.model_names = []
        self.current_model = None
        self.models_info = []

        model_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        for model_dir in model_dirs:
            model_files = [
                f
                for f in model_dir.iterdir()
                if f.suffix in [".pth", ".pt", ".safetensors"]
            ]
            if len(model_files) == 0:
                logger.warning(f"{model_dir}에서 모델 파일을 찾을 수 없습니다.")
                continue
            config_path = model_dir / "config.json"
            if not config_path.exists():
                logger.warning(
                    f"{config_path}가 존재하지 않으므로 {model_dir}를 건너뜁니다."
                )
                continue
            self.model_files_dict[model_dir.name] = model_files
            self.model_names.append(model_dir.name)
            hyper_parameters = HyperParameters.load_from_json(config_path)
            style2id: dict[str, int] = hyper_parameters.data.style2id
            styles = list(style2id.keys())
            spk2id: dict[str, int] = hyper_parameters.data.spk2id
            speakers = list(spk2id.keys())
            self.models_info.append(
                TTSModelInfo(
                    name=model_dir.name,
                    files=[str(f) for f in model_files],
                    styles=styles,
                    speakers=speakers,
                )
            )

    def get_model(self, model_name: str, model_path_str: str) -> TTSModel:
        model_path = Path(model_path_str)
        if model_name not in self.model_files_dict:
            raise ValueError(f"모델 `{model_name}`이(가) 발견되지 않았습니다.")
        if model_path not in self.model_files_dict[model_name]:
            raise ValueError(f"모델 파일 `{model_path}`이(가) 발견되지 않았습니다.")
        if self.current_model is None or self.current_model.model_path != model_path:
            self.current_model = TTSModel(
                model_path=model_path,
                config_path=self.root_dir / model_name / "config.json",
                style_vec_path=self.root_dir / model_name / "style_vectors.npy",
                device=self.device,
            )

        return self.current_model