# ====================================================================
#  File: services/tts.py
# ====================================================================
"""
L.U.N.A. TTS 합성 서비스 모듈

 - luna_core.tts_model.TTSModelHolder를 사용하여 실제 음성 합성을 실행
 - 합성된 오디오는 outputs 디렉토리에 WAV로 저장되고, 그 URL을 반환
 
    2025/07/13
     - 예외처리 추가 및 전체 코드 옵티마이징
"""

from __future__ import annotations

import os
import uuid
import hashlib
import json
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
import soundfile as sf
from fastapi import HTTPException
from utils.config import load_config_dict
from utils.style_map import get_style_from_emotion, get_top_emotion
from services.emotion import EmotionService
from models.tts_model import TTSModelHolder


# ─────────────────────────────────────────────────────────────────────
# 예외 처리
# ─────────────────────────────────────────────────────────────────────
class TTSError(RuntimeError):
    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code, self.detail = code, detail

# ─────────────────────────────────────────────────────────────────────
# TTS 캐시 클래스
# ─────────────────────────────────────────────────────────────────────
class TTSCache:
    """
    TTS 오디오 캐싱 시스템
    
    - 해시 기반 캐싱 (텍스트 + 스타일 + 파라미터)
    - 파일 시스템 기반 저장
    - LRU 정책 지원
    - TTL (Time To Live) 지원
    """
    
    def __init__(
        self,
        cache_dir: str = "./cache/tts",
        max_cache_size: int = 100,
        ttl: int = 3600 * 24 * 7  # 7일
    ):
        """
        Args:
            cache_dir: 캐시 파일 저장 디렉토리
            max_cache_size: 최대 캐시 항목 수
            ttl: 캐시 유효 시간 (초)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size = max_cache_size
        self.ttl = ttl
        
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()
        
        # 통계
        self.hits = 0
        self.misses = 0
        
        print(f"[TTSCache] 초기화 완료 (캐시 크기: {len(self.metadata)}, 최대: {max_cache_size})")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """메타데이터 로드"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[TTSCache] 메타데이터 로드 실패: {e}")
        return {}
    
    def _save_metadata(self):
        """메타데이터 저장"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[TTSCache] 메타데이터 저장 실패: {e}")
    
    def _generate_key(
        self,
        text: str,
        style: str,
        style_weight: float,
        noise_scale: float,
        noise_scale_w: float,
        length_scale: float
    ) -> str:
        """캐시 키 생성 (해시)"""
        key_str = f"{text}|{style}|{style_weight}|{noise_scale}|{noise_scale_w}|{length_scale}"
        return hashlib.sha256(key_str.encode('utf-8')).hexdigest()
    
    def get(
        self,
        text: str,
        style: str,
        style_weight: float,
        noise_scale: float,
        noise_scale_w: float,
        length_scale: float
    ) -> Optional[str]:
        """
        캐시에서 오디오 파일 경로 가져오기
        
        Returns:
            str: 오디오 파일 경로 (캐시 미스 시 None)
        """
        key = self._generate_key(text, style, style_weight, noise_scale, noise_scale_w, length_scale)
        
        if key not in self.metadata:
            self.misses += 1
            return None
        
        entry = self.metadata[key]
        
        # TTL 확인
        if time.time() - entry['timestamp'] > self.ttl:
            self.misses += 1
            self._remove_entry(key)
            return None
        
        # 파일 존재 확인
        audio_path = self.cache_dir / entry['filename']
        if not audio_path.exists():
            self.misses += 1
            self._remove_entry(key)
            return None
        
        # LRU 업데이트
        entry['last_accessed'] = time.time()
        entry['access_count'] += 1
        self._save_metadata()
        
        self.hits += 1
        return str(audio_path)
    
    def set(
        self,
        text: str,
        style: str,
        style_weight: float,
        noise_scale: float,
        noise_scale_w: float,
        length_scale: float,
        audio_path: str
    ):
        """캐시에 오디오 저장"""
        key = self._generate_key(text, style, style_weight, noise_scale, noise_scale_w, length_scale)
        
        # LRU: 캐시 크기 초과 시 가장 오래된 항목 삭제
        if len(self.metadata) >= self.max_cache_size:
            self._evict_oldest()
        
        # 메타데이터 저장
        self.metadata[key] = {
            'filename': Path(audio_path).name,
            'text': text[:100],  # 처음 100자만 저장
            'style': style,
            'style_weight': style_weight,
            'timestamp': time.time(),
            'last_accessed': time.time(),
            'access_count': 1
        }
        
        self._save_metadata()
    
    def _remove_entry(self, key: str):
        """캐시 항목 삭제"""
        if key in self.metadata:
            try:
                filename = self.metadata[key]['filename']
                audio_path = self.cache_dir / filename
                if audio_path.exists():
                    audio_path.unlink()
            except Exception as e:
                print(f"[TTSCache] 파일 삭제 실패: {e}")
            
            del self.metadata[key]
            self._save_metadata()
    
    def _evict_oldest(self):
        """가장 오래 접근되지 않은 항목 삭제 (LRU)"""
        if not self.metadata:
            return
        
        oldest_key = min(
            self.metadata.keys(),
            key=lambda k: self.metadata[k]['last_accessed']
        )
        
        print(f"[TTSCache] LRU 삭제: {self.metadata[oldest_key]['text'][:50]}...")
        self._remove_entry(oldest_key)
    
    def cleanup_expired(self) -> int:
        """만료된 캐시 항목 삭제"""
        expired_keys = []
        current_time = time.time()
        
        for key, entry in self.metadata.items():
            if current_time - entry['timestamp'] > self.ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_entry(key)
        
        return len(expired_keys)
    
    def clear(self):
        """모든 캐시 삭제"""
        for key in list(self.metadata.keys()):
            self._remove_entry(key)
        
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_size': len(self.metadata),
            'max_cache_size': self.max_cache_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': round(hit_rate, 2),
            'total_requests': total_requests
        }

# ─────────────────────────────────────────────────────────────────────
# TTSService 클래스
# ─────────────────────────────────────────────────────────────────────
class TTSService:
    """
    L.U.N.A. TTS 합성 서비스 클래스
    
    텍스트 -> 음성 합성
    """
    
    _holder: Optional[TTSModelHolder] = None
    _model: Optional[torch.nn.Module] = None
    _sampling_rate: Optional[int] = None
    _config: Optional[dict] = None
    _is_warmed_up: bool = False

    def __init__(self, device: str, enable_cache: bool = True):
        tts_config = load_config_dict("models")["tts"]        
        self.device = device
        
        model_dir = Path(tts_config["model_dir"]).expanduser()
        self.default_name = tts_config.get("default_model")
        self.outputs_dir = Path(tts_config.get("output_dir", "outputs"))
        self.outputs_dir.mkdir(exist_ok=True)
        
        # 캐시 초기화
        self.enable_cache = enable_cache
        if enable_cache:
            cache_config = tts_config.get("cache", {})
            self.cache = TTSCache(
                cache_dir=cache_config.get("cache_dir", "./cache/tts"),
                max_cache_size=cache_config.get("max_cache_size", 100),
                ttl=cache_config.get("ttl", 3600 * 24 * 7)
            )
        else:
            self.cache = None
        
        # Thread pool for parallel segment processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        if TTSService._holder is None or TTSService._config != tts_config:
            TTSService._config = tts_config
            TTSService._holder = TTSModelHolder(str(model_dir), device)
            self._load_default_model(tts_config)

        TTSService._sampling_rate = tts_config.get("sampling_rate", 44100)
        self.noise_scale = tts_config.get("noise_scale", 0.6)
        self.noise_scale_w = tts_config.get("noise_scale_w", 0.8)
        self.length_scale = tts_config.get("length_scale", 1.0)
        self.sdp_ratio = tts_config.get("sdp_ratio", 0.2)
        self.split_interval = tts_config.get("split_interval", 1.0)
        self.line_split = tts_config.get("line_split", False)

        self.emotion_service = EmotionService()

    def _load_default_model(self, tts_config: dict) -> None:
        """TTSModelHolder -> Default_Model 캐싱 (torch.compile 제거)"""
        try:
            paths = TTSService._holder.model_files_dict[self.default_name]
        except KeyError:
            raise TTSError(
                "TTS_MODEL_NOT_FOUND",
                f"기본 TTS 모델 '{self.default_name}'을 찾을 수 없습니다."
            )
        
        TTSService._holder.get_model(self.default_name, str(paths[0]))
        model = TTSService._holder.current_model
        if model is None:
            raise TTSError(
                "TTS_MODEL_LOAD_ERROR",
                f"기본 TTS 모델 '{self.default_name}'을 로드할 수 없습니다."
            )
            
        if isinstance(model, torch.nn.Module):
            # if torch.cuda.is_available() and torch.__version__ >= "2.0.0":
            #     model = torch.compile(model, mode="default")
            model.eval()
            
        TTSService._model = model
        print(f"[TTS] 모델 '{self.default_name}' 로드 완료")
    
    def warmup(self) -> None:
        """웜업 추론 실행"""
        if TTSService._is_warmed_up:
            print("[TTS] 이미 웜업 완료됨")
            return
            
        print("[TTS] 웜업 추론 시작...")
        try:
            model = TTSService._model
            if model is None:
                raise TTSError("TTS_MODEL_NOT_LOADED", "TTS 모델이 로드되지 않았습니다.")
            
            with torch.autocast(self.device, torch.float16, enabled=self.device == "cuda"):
                sr, audio = model.infer(
                    text="こんにちは",
                    language="JP",
                    reference_audio_path=None,
                    sdp_ratio=self.sdp_ratio,
                    noise=self.noise_scale,
                    noise_w=self.noise_scale_w,
                    length=self.length_scale,
                    line_split=False,
                    split_interval=1.0,
                    assist_text=None,
                    assist_text_weight=0.0,
                    use_assist_text=False,
                    style="Neutral",
                    style_weight=1.0,
                    given_tone=None,
                    speaker_id=list(model.spk2id.values())[0],
                    pitch_scale=1.0,
                    intonation_scale=1.0,
                )
            TTSService._is_warmed_up = True
            print("[TTS] 웜업 추론 완료!")
        except Exception as e:
            print(f"[TTS] 웜업 실패: {e}")
    
    @torch.inference_mode()
    def synthesize(
        self,
        text: str,
        style: Optional[str] = None,
        style_weight: Optional[float] = None,
        use_emotion_analysis: bool = True,
        use_cache: bool = True
    ) -> dict:
        """
        주어진 텍스트를 TTS 모델을 사용하여 음성으로 합성하고,
        음성 파일의 URL과 스타일 정보를 반환합니다.
        
        Args:
            text (str): 합성할 텍스트
            style (str, optional): 스타일 (지정 시 감정 분석 생략 가능)
            style_weight (float, optional): 스타일 가중치
            use_emotion_analysis (bool): 감정 분석 사용 여부
            use_cache (bool): 캐싱 사용 여부
            
        Returns:
            dict: 음성 파일의 URL과 스타일 정보
        """
        if not text or not text.strip():
            raise TTSError("TTS_EMPTY_INPUT", "입력 텍스트가 비어 있습니다.")
        
        # 감정 분석 (필요시만)
        top_emotion = None
        if use_emotion_analysis and (style is None or style_weight is None):
            emotion_scores = self.emotion_service.predict(text)
            top_emotion = get_top_emotion(emotion_scores)
        
        if style is None or style_weight is None:
            if top_emotion:
                style, style_weight = get_style_from_emotion(top_emotion)
            else:
                style, style_weight = "Neutral", 1.0
        
        # 캐시 확인
        if use_cache and self.cache:
            cached_path = self.cache.get(
                text=text,
                style=style,
                style_weight=style_weight,
                noise_scale=self.noise_scale,
                noise_scale_w=self.noise_scale_w,
                length_scale=self.length_scale
            )
            
            if cached_path:
                filename = Path(cached_path).name
                return {
                    "audio_url": f"/outputs/{filename}",
                    "emotion": top_emotion or "cached",
                    "style": style,
                    "style_weight": style_weight,
                    "cached": True
                }
        
        model = TTSService._model
        if model is None:
            raise TTSError("TTS_MODEL_NOT_LOADED", "TTS 모델이 로드되지 않았습니다.")
        
        try:
            segments: List[str] = [text]
            if self.line_split or len(text) > 200:
                segments = [
                    text[i : i + int(self.split_interval * 100)]
                    for i in range(0, len(text), int(self.split_interval * 100))
                ]
            
            audios = []
            for seg in segments:
                with torch.autocast(self.device, torch.float16, enabled=self.device == "cuda"):
                    sr, audio = model.infer(
                        text=seg,
                        language="JP",
                        reference_audio_path=None,
                        sdp_ratio=self.sdp_ratio,
                        noise=self.noise_scale,
                        noise_w=self.noise_scale_w,
                        length=self.length_scale,
                        line_split=self.line_split,
                        split_interval=self.split_interval,
                        assist_text=None,
                        assist_text_weight=0.0,
                        use_assist_text=False,
                        style=style,
                        style_weight=style_weight,
                        given_tone=None,
                        speaker_id=list(model.spk2id.values())[0],
                        pitch_scale=1.0,
                        intonation_scale=1.0,
                    )
                audios.append(audio)
                
            full_audio = torch.cat([torch.from_numpy(a) for a in audios]).numpy()
            
        except Exception as e:
            raise TTSError("TTS_INFERENCE_ERROR", f"TTS 합성 중 오류 발생: {str(e)}") from e

        filename = f"{uuid.uuid4()}.wav"
        outpath = self.outputs_dir / filename
        try:
            sf.write(str(outpath), full_audio, TTSService._sampling_rate)
        except Exception as e:
            raise TTSError("TTS_FILE_WRITE_ERROR", f"합성된 오디오 파일 저장 실패: {str(e)}") from e

        # 캐시에 저장
        if use_cache and self.cache:
            self.cache.set(
                text=text,
                style=style,
                style_weight=style_weight,
                noise_scale=self.noise_scale,
                noise_scale_w=self.noise_scale_w,
                length_scale=self.length_scale,
                audio_path=str(outpath)
            )

        return {
            "audio_url": f"/outputs/{filename}",
            "emotion": top_emotion,
            "style": style,
            "style_weight": style_weight,
            "cached": False
        }
    
    async def synthesize_async(
        self,
        text: str,
        style: Optional[str] = None,
        style_weight: Optional[float] = None,
        use_emotion_analysis: bool = True,
        use_cache: bool = True
    ) -> dict:
        """
        비동기 TTS 합성 (기존 동기 메서드를 비동기로 래핑)
        
        Args:
            text (str): 합성할 텍스트
            style (str, optional): 스타일
            style_weight (float, optional): 스타일 가중치
            use_emotion_analysis (bool): 감정 분석 사용 여부
            use_cache (bool): 캐싱 사용 여부
            
        Returns:
            dict: 음성 파일의 URL과 스타일 정보
        """
        # 동기 메서드를 스레드에서 실행
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: self.synthesize(
                text=text,
                style=style,
                style_weight=style_weight,
                use_emotion_analysis=use_emotion_analysis,
                use_cache=use_cache
            )
        )
    
    def _synthesize_segment(
        self,
        segment: str,
        style: str,
        style_weight: float,
        model: torch.nn.Module
    ) -> np.ndarray:
        """
        단일 세그먼트 합성 (병렬 처리용 헬퍼 메서드)
        
        Args:
            segment: 텍스트 세그먼트
            style: 스타일
            style_weight: 스타일 가중치
            model: TTS 모델
            
        Returns:
            np.ndarray: 합성된 오디오
        """
        with torch.autocast(self.device, torch.float16, enabled=self.device == "cuda"):
            sr, audio = model.infer(
                text=segment,
                language="JP",
                reference_audio_path=None,
                sdp_ratio=self.sdp_ratio,
                noise=self.noise_scale,
                noise_w=self.noise_scale_w,
                length=self.length_scale,
                line_split=self.line_split,
                split_interval=self.split_interval,
                assist_text=None,
                assist_text_weight=0.0,
                use_assist_text=False,
                style=style,
                style_weight=style_weight,
                given_tone=None,
                speaker_id=list(model.spk2id.values())[0],
                pitch_scale=1.0,
                intonation_scale=1.0,
            )
        return audio
    
    async def synthesize_parallel(
        self,
        text: str,
        style: Optional[str] = None,
        style_weight: Optional[float] = None,
        use_emotion_analysis: bool = True,
        use_cache: bool = True,
        max_parallel: int = 4
    ) -> dict:
        """
        병렬 처리를 사용한 TTS 합성 (긴 텍스트에 효과적)
        
        Args:
            text (str): 합성할 텍스트
            style (str, optional): 스타일
            style_weight (float, optional): 스타일 가중치
            use_emotion_analysis (bool): 감정 분석 사용 여부
            use_cache (bool): 캐싱 사용 여부
            max_parallel (int): 최대 병렬 세그먼트 수
            
        Returns:
            dict: 음성 파일의 URL과 스타일 정보
        """
        if not text or not text.strip():
            raise TTSError("TTS_EMPTY_INPUT", "입력 텍스트가 비어 있습니다.")
        
        # 짧은 텍스트는 일반 메서드 사용
        if len(text) <= 200:
            return await self.synthesize_async(
                text=text,
                style=style,
                style_weight=style_weight,
                use_emotion_analysis=use_emotion_analysis,
                use_cache=use_cache
            )
        
        # 감정 분석
        top_emotion = None
        if use_emotion_analysis and (style is None or style_weight is None):
            emotion_scores = self.emotion_service.predict(text)
            top_emotion = get_top_emotion(emotion_scores)
        
        if style is None or style_weight is None:
            if top_emotion:
                style, style_weight = get_style_from_emotion(top_emotion)
            else:
                style, style_weight = "Neutral", 1.0
        
        # 캐시 확인
        if use_cache and self.cache:
            cached_path = self.cache.get(
                text=text,
                style=style,
                style_weight=style_weight,
                noise_scale=self.noise_scale,
                noise_scale_w=self.noise_scale_w,
                length_scale=self.length_scale
            )
            
            if cached_path:
                filename = Path(cached_path).name
                return {
                    "audio_url": f"/outputs/{filename}",
                    "emotion": top_emotion or "cached",
                    "style": style,
                    "style_weight": style_weight,
                    "cached": True
                }
        
        # 모델 가져오기
        model = TTSService._model
        if model is None:
            raise TTSError("TTS_MODEL_NOT_LOADED", "TTS 모델이 로드되지 않았습니다.")
        
        # 세그먼트 분할
        segments = [
            text[i : i + int(self.split_interval * 100)]
            for i in range(0, len(text), int(self.split_interval * 100))
        ]
        
        # 병렬 처리
        loop = asyncio.get_event_loop()
        
        tasks = []
        for segment in segments:
            task = loop.run_in_executor(
                self.executor,
                self._synthesize_segment,
                segment,
                style,
                style_weight,
                model
            )
            tasks.append(task)
        
        # 모든 세그먼트 합성 완료 대기
        audios = await asyncio.gather(*tasks)
        
        # 오디오 결합
        full_audio = torch.cat([torch.from_numpy(a) for a in audios]).numpy()
        
        # 파일 저장
        filename = f"{uuid.uuid4()}.wav"
        outpath = self.outputs_dir / filename
        
        try:
            sf.write(str(outpath), full_audio, TTSService._sampling_rate)
        except Exception as e:
            raise TTSError("TTS_FILE_WRITE_ERROR", f"합성된 오디오 파일 저장 실패: {str(e)}") from e
        
        # 캐시에 저장
        if use_cache and self.cache:
            self.cache.set(
                text=text,
                style=style,
                style_weight=style_weight,
                noise_scale=self.noise_scale,
                noise_scale_w=self.noise_scale_w,
                length_scale=self.length_scale,
                audio_path=str(outpath)
            )
        
        return {
            "audio_url": f"/outputs/{filename}",
            "emotion": top_emotion,
            "style": style,
            "style_weight": style_weight,
            "cached": False,
            "parallel": True,
            "segments": len(segments)
        }