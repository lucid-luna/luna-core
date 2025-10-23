# ====================================================================
#  File: services/interaction.py
# ====================================================================
import json
import logging
import asyncio
from pathlib import Path
from pydantic import BaseModel

from .emotion import EmotionService
from .multi_intent import MultiIntentService
from .translator import TranslatorService
from .llm_manager import LLMManager
from .tts import TTSService
from .mcp.tool_registry import ToolRegistry
from .memory import MemoryService
from utils.style_map import get_style_from_emotion

class InteractResponse(BaseModel):
    text: str
    emotion: str
    intent: str
    style: str
    audio_url: str

class InteractionService:
    def __init__(
        self,
        emotion_service: EmotionService,
        multi_intent_service: MultiIntentService,
        translator_service: TranslatorService,
        llm_service: LLMManager,
        tts_service: TTSService,
        tool_registry: ToolRegistry,
        memory_service: MemoryService = None,
        llm_target: str = "rp",
        prompt_dir: str = "./checkpoints/LunaLLM",
        logger: logging.Logger = None
    ):
        self.emotion_service = emotion_service
        self.multi_intent_service = multi_intent_service
        self.translator_service = translator_service
        self.llm_service = llm_service
        self.tts_service = tts_service
        self.tool_registry = tool_registry
        self.memory_service = memory_service or MemoryService()
        self.llm_target = llm_target
        self.logger = logger

        try:
            prompt_path = Path(prompt_dir)
            # API 모드일 때는 한국어 프롬프트 사용
            if llm_service.get_mode() == "api":
                self.rp_prompt_template = (prompt_path / "prompt_Kurumi_kr.txt").read_text(encoding='utf-8')
                print("[InteractionService] API 모드: 한국어 프롬프트(prompt_Kurumi_kr.txt) 로딩 완료.")
            else:
                self.rp_prompt_template = (prompt_path / "prompt_Kurumi.txt").read_text(encoding='utf-8')
                print("[InteractionService] 서버 모드: 영문 프롬프트(prompt_Kurumi.txt) 로딩 완료.")
            
            self.trans_prompt_template = (prompt_path / "prompt_Translate.txt").read_text(encoding='utf-8')
        except Exception as e:
            print(f"[InteractionService] 프롬프트 로딩 실패: {e}")
            self.rp_prompt_template = "User: {user_input}\nAssistant:"
            self.trans_prompt_template = "Translate the following text to Korean."

    def run(self, ko_input_text: str) -> InteractResponse:
        self.logger.info(f"[InteractionService] 사용자 입력: {ko_input_text}")
        
        music_keyworks = ["노래", "음악", "뮤직", "틀어줘", "재생", "플레이", "멈춰", "일시정지", "다음곡", "넘겨줘", "스킵"]
        if any(keyword in ko_input_text for keyword in music_keyworks):
            self.logger.info("[InteractionService] 음악 관련 요청 감지, 도구 사용 시도.")
            return self._run_agent_mode(ko_input_text)
        else:
            self.logger.info("[InteractionService] 일반 대화 요청 감지, 일반 모드로 처리.")
            return self._run_normal_mode(ko_input_text)
        
    def _run_agent_mode(self, ko_input_text: str) -> InteractResponse:
        en_input = self.translator_service.translate(ko_input_text, "ko", "en")
            
        messages = [
            {"role": "system", "content": self.rp_prompt_template},
            {"role": "user", "content": en_input}
        ]
        
        available_tools = self.tool_registry.get_all_tool_definitions()
        
        for _ in range(5):
            llm_response = self.llm_service.generate(
                target=self.llm_target,
                messages=messages,
                tools=available_tools,
            )
            if not llm_response or "choices" not in llm_response:
                return self.error_response("LLM 응답 처리 중 오류가 발생했습니다.")
            
            message = llm_response["choices"][0]["message"]
            messages.append(message)
            
            if not message.get("tool_calls"):
                break
            
            tool_calls = message["tool_calls"]
            for tool_call in tool_calls:
                name, args = tool_call["function"]["name"], json.loads(tool_call["function"]["arguments"])
                self.logger.info(f"에이전트 도구 호출: {name}({args})")
                try:
                    result = self.tool_registry.dispatch_tool_call(name, args)
                    messages.append({"role": "tool", "tool_call_id": tool_call["id"], "name": name, "content": str(result)})
                except Exception as e:
                    self.logger.error(f"Tool 실행 오류: {e}", exc_info=True)
                    messages.append({"role": "tool", "tool_call_id": tool_call["id"], "name": name, "content": f"Error: {e}"})
        
        en_final_text = messages[-1].get("content", "Failed to generate a final response.")
        
        # trans_response_dict = self.llm_service.generate(
        #     system_prompt=self.trans_prompt_template,
        #     user_prompt=en_final_text,
        #     target_server="translator"
        # )
        # ko_final_text = trans_response_dict["choices"][0]["message"]["content"].strip()
        
        ko_final_text = self.translator_service.translate(en_final_text, "en", "ko")
        
        top_emotion = "neutral"
        style, style_weight = get_style_from_emotion(top_emotion)
        
        ja_text_for_tts = self.translator_service.translate(ko_final_text, "ko", "ja")
        tts_result = self.tts_service.synthesize(text=ja_text_for_tts, style=style, style_weight=style_weight)
        
        return InteractResponse(
            text=ko_final_text, emotion=top_emotion, intent="agent_spotify",
            style=style, audio_url=tts_result.get("audio_url", "")
        )
        
    def _run_normal_mode(self, ko_input_text: str) -> InteractResponse:
        try:
            is_api_mode = self.llm_service.get_mode() == "api"
            
            # API 모드(한국어 프롬프트)일 때는 번역 건너뛰기
            if is_api_mode:
                input_text = ko_input_text
                self.logger.info(f"[API 모드] 한국어 입력 사용: {input_text}")
            else:
                # 서버 모드(영문 프롬프트)일 때는 영어로 번역
                input_text = self.translator_service.translate(ko_input_text, "ko", "en")
                self.logger.info(f"[서버 모드] 영어로 번역: {input_text}")
            
            emotion_probs = self.emotion_service.predict(input_text)
            top_emotion = max(emotion_probs, key=emotion_probs.get) if emotion_probs else "neutral"

            # 메모리에서 최근 대화 컨텍스트 가져오기
            context_messages = self.memory_service.get_context_for_llm()
            
            messages = [
                {"role": "system", "content": self.rp_prompt_template}
            ]
            
            # 기존 대화 컨텍스트 추가
            messages.extend(context_messages)
            
            # 현재 사용자 입력 추가
            messages.append({"role": "user", "content": input_text})

            llm_response_dict = self.llm_service.generate(
                target=self.llm_target,
                messages=messages
            )
            response_text = llm_response_dict["choices"][0]["message"]["content"].strip()
            self.logger.info(f"LLM 응답: {response_text}")

            # API 모드일 때는 이미 한국어이므로 번역 건너뛰기
            if is_api_mode:
                ko_response = response_text
                self.logger.info(f"[API 모드] 한국어 응답 사용: {ko_response}")
            else:
                # 서버 모드일 때는 한국어로 번역
                ko_response = self.translator_service.translate(response_text, "en", "ko")
                self.logger.info(f"[서버 모드] 한국어로 번역: {ko_response}")
            
            # 메모리에 대화 저장
            self.memory_service.add_entry(
                user_input=ko_input_text,
                assistant_response=ko_response,
                metadata={
                    "emotion": top_emotion,
                    "intent": "general",
                    "mode": "api" if is_api_mode else "server"
                }
            )
            self.logger.info(f"[Memory] 대화 저장 완료")
            
            ja_text_for_tts = self.translator_service.translate(ko_response, "ko", "ja")
            style, style_weight = get_style_from_emotion(top_emotion)
            
            # TTS를 비동기로 실행
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            tts_result = loop.run_until_complete(
                self.tts_service.synthesize_async(
                    text=ja_text_for_tts,
                    style=style,
                    style_weight=style_weight
                )
            )

            return InteractResponse(
                text=ko_response, emotion=top_emotion, intent="general",
                style=style, audio_url=tts_result.get("audio_url", "")
            )
        except Exception as e:
            self.logger.error(f"파이프라인 실행 중 오류: {e}", exc_info=True)
            return self.error_response("일반 응답 처리 중 오류가 발생했습니다.")

    def error_response(self, error_message: str) -> InteractResponse:
        self.logger.error(f"오류 응답 반환: {error_message}")
        return InteractResponse(text=f"오류: {error_message}", emotion="neutral", intent="error", style="Neutral", audio_url="")
