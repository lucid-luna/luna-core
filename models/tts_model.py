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
from models.tts.models_jp_extra import SynthesizerTrn
from models.voice import adjust_voice

logger = logging.getLogger(__name__)

class TTSModel:
    """
    LunaTTS의 음성 합성 모델을 초기화하고 음성 합성을 수행하는 클래스입니다.
    모델/하이퍼파라미터/스타일 벡터의 경로와 장치를 지정하여 초기화하고, model.infer() 메서드를 호출하면 음성 합성을 수행할 수 있습니다.
    """
    
    def __init__(
        self,
        model_path: Path,
        config_path: Union[Path, HyperParameters],
        style_vec_path: Union[Path, NDArray[Any]],
        device: str,
    ) -> None:
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
                f"Number of styles ({num_styles}) does not match the number of style2id ({len(self.style2id)})"
            )

        if self.__style_vectors.shape[0] != num_styles:
            raise ValueError(
                f"The number of styles ({num_styles}) does not match the number of style vectors ({self.__style_vectors.shape[0]})"
            )
        self.__style_vector_inference: Optional[Any] = None

        self.__net_g: Union[SynthesizerTrn, None] = None
    
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
                    "pyannote.audio is required to infer style vector from audio"
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
        if data.dtype in [np.float64, np.float32, np.float16]:
            data = data / np.abs(data).max() * 32767
            return data.astype(np.int16)
        elif data.dtype == np.int32:
            return (data / 65536).astype(np.int16)
        elif data.dtype == np.uint16:
            return (data - 32768).astype(np.int16)
        elif data.dtype == np.uint8:
            return (data * 257 - 32768).astype(np.int16)
        elif data.dtype == np.int8:
            return (data * 256).astype(np.int16)
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

        logger.info(f"[L.U.N.A.] Start generating audio data from text:\n{text}")
        if language != "JP" and self.hyper_parameters.version.endswith("JP-Extra"):
            raise ValueError(
                "The model is trained with JP-Extra, but the language is not JP"
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
        logger.info("Audio data generated successfully")
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
    Style-Bert-Vits2의 음성 합성 모델을 관리하는 클래스입니다.
    model_holder.models_info 에서 모델 정보를 가져오고, model_holder.get_model() 메서드를 통해 모델을 로드합니다.
    """
    def __init__(self, model_root_dir: Path, device: str) -> None:
        self.root_dir: Path = model_root_dir
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

        for model_dir in [d for d in self.root_dir.iterdir() if d.is_dir()]:
            model_files = [f for f in model_dir.iterdir() if f.suffix in [".pth", ".pt", ".safetensors"]]
            if not model_files:
                continue
            config_path = model_dir / "config.json"
            if not config_path.exists():
                continue
            self.model_files_dict[model_dir.name] = model_files
            self.model_names.append(model_dir.name)
            hp = HyperParameters.load_from_json(config_path)
            styles = list(hp.data.style2id.keys())
            speakers = list(hp.data.spk2id.keys())
            self.models_info.append(TTSModelInfo(name=model_dir.name, files=[str(f) for f in model_files], styles=styles, speakers=speakers))

    def get_model(self, model_name: str, model_path_str: str) -> TTSModel:
        model_path = Path(model_path_str)
        if model_name not in self.model_files_dict:
            raise ValueError(f"Model {model_name} not found")
        if model_path not in self.model_files_dict[model_name]:
            raise ValueError(f"Model file {model_path} not found")
        if self.current_model is None or self.current_model.model_path != model_path:
            self.current_model = TTSModel(
                model_path=model_path,
                config_path=self.root_dir / model_name / "config.json",
                style_vec_path=self.root_dir / model_name / "style_vectors.npy",
                device=self.device,
            )
        return self.current_model
