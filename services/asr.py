# ====================================================================
#  File: services/asr.py
# ====================================================================
"""
ASR 서비스 모듈
"""

import os
import asyncio
import azure.cognitiveservices.speech as speechsdk
from typing import Callable
from azure.cognitiveservices.speech.audio import AudioStreamFormat

class ASRService:
    def __init__(self):
        """ASR 서비스 초기화 및 API 설정"""
        speech_key = os.environ.get("AZURE_SPEECH_KEY")
        speech_region = os.environ.get("AZURE_SPEECH_REGION")

        if not speech_key or not speech_region:
            raise ValueError("[L.U.N.A.] Azure Speech API 키 또는 지역이 설정되지 않았습니다.")
        
        self.speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
        self.speech_config.speech_recognition_language = "ko-KR"
        self.speech_config.output_format = speechsdk.OutputFormat.Detailed
        
        self.push_stream = None
        self.recognizer = None
        
    def transcribe_stream(self, result_callback: Callable[[str], None]):
        """
        음성 스트림을 받아 텍스트로 변환하고 콜백 함수로 결과를 전달합니다.

        Args:
            result_callback (Callable[[str], None]): 변환된 텍스트 결과를 처리하는 콜백 함수
        """
        stream_format = AudioStreamFormat(
            samples_per_second=44100,
            bits_per_sample=16,
            channels=1
        )
        self.push_stream = speechsdk.audio.PushAudioInputStream(stream_format)
        
        audio_config = speechsdk.audio.AudioConfig(stream=self.push_stream)
        self.recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config, audio_config=audio_config)
        
        main_loop = asyncio.get_running_loop()
        
        def recognized_cb(evt: speechsdk.SpeechRecognitionEventArgs):
            result_text = ""
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                result_text = evt.result.text
            
            if result_text:
                print(f"[L.U.N.A. ASR Service] 인식된 텍스트: {result_text}")
                asyncio.run_coroutine_threadsafe(result_callback(result_text), main_loop)

        
        def canceled_cb(evt: speechsdk.SpeechRecognitionCanceledEventArgs):
            print(f"[L.U.N.A. ASR Service] 음성 인식 취소: {evt.reason}")

        self.recognizer.recognized.connect(recognized_cb)
        self.recognizer.canceled.connect(canceled_cb)
        self.recognizer.session_stopped.connect(lambda evt: print(f"[L.U.N.A. ASR Service] 음성 인식 세션 중지: {evt.reason}"))
        
        self.recognizer.start_continuous_recognition_async()
        
        return self.push_stream
    
    def stop_transcription(self):
        """음성 인식 중지 및 리소스 해제"""
        if self.recognizer:
            self.recognizer.stop_continuous_recognition_async()
            self.recognizer = None
        if self.push_stream:
            self.push_stream.close()
            self.push_stream = None
        print("[L.U.N.A. ASR Service] 음성 인식 중지 및 리소스 해제 완료.")