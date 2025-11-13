# ====================================================================
#  File: services/interaction.py
# ====================================================================
import json
import logging
import asyncio
import re
import os
from pathlib import Path
from pydantic import BaseModel

from .emotion import EmotionService
from .multi_intent import MultiIntentService
from .translator import TranslatorService
from .llm_manager import LLMManager
from .tts import TTSService
from .mcp.tool_registry import ToolRegistry
from .mcp.tool_manager import MCPToolManager
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
        mcp_tool_manager: MCPToolManager = None,
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
        self.mcp_tool_manager = mcp_tool_manager
        self.memory_service = memory_service or MemoryService()
        self.llm_target = llm_target
        self.logger = logger
        
        self._mcp_tools_cache = None
        self._mcp_tools_cache_time = 0

        try:
            prompt_path = Path(prompt_dir)
            # API 모드일 때는 한국어 프롬프트 사용
            if llm_service.get_mode() == "api":
                self.rp_prompt_template = (prompt_path / "prompt_Kurumi_kr.txt").read_text(encoding='utf-8')
                print("[L.U.N.A. InteractionService] API 모드: 한국어 프롬프트(prompt_Kurumi_kr.txt) 로딩 완료.")
            else:
                self.rp_prompt_template = (prompt_path / "prompt_Kurumi.txt").read_text(encoding='utf-8')
                print("[L.U.N.A. InteractionService] 로컬 서버 모드: 영문 프롬프트(prompt_Kurumi.txt) 로딩 완료.")
            
            self.trans_prompt_template = (prompt_path / "prompt_Translate.txt").read_text(encoding='utf-8')
        except Exception as e:
            print(f"[L.U.N.A. InteractionService] 프롬프트 로딩 실패: {e}")
            self.rp_prompt_template = "User: {user_input}\nAssistant:"
            self.trans_prompt_template = "Translate the following text to Korean."

    def _get_available_mcp_tools(self) -> list:
        """MCP 도구 목록 조회 (캐싱 적용)"""
        import time
        
        if not self.mcp_tool_manager:
            return []
        
        current_time = time.time()
        if self._mcp_tools_cache is not None and (current_time - self._mcp_tools_cache_time) < 30:
            return self._mcp_tools_cache
        
        try:
            mcp_tools = self.mcp_tool_manager.get_tool_list()
            self._mcp_tools_cache = mcp_tools if mcp_tools else []
            self._mcp_tools_cache_time = current_time
            return self._mcp_tools_cache
        except Exception as e:
            self.logger.warning(f"[L.U.N.A. InteractionService] 도구 목록 조회 실패: {e}")
            return []
    
    def _call_mcp_tool(self, server_id: str, tool_name: str, arguments: dict, timeout: float | None = None) -> str:
        if not self.mcp_tool_manager:
            raise Exception("MCP 도구 매니저가 없습니다.")
        
        if timeout is None:
            timeout = float(os.getenv("LUNA_TOOL_TIMEOUT", "15"))
        
        try:
            import time
            
            if not hasattr(self, '_event_loop'):
                try:
                    self._event_loop = asyncio.get_event_loop()
                    if self._event_loop.is_closed():
                        self._event_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(self._event_loop)
                except RuntimeError:
                    self._event_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(self._event_loop)

            self.logger.debug(f"[L.U.N.A. InteractionService] 도구 호출 상세 정보:")
            self.logger.debug(f"  - Server: {server_id}")
            self.logger.debug(f"  - Tool: {tool_name}")
            self.logger.debug(f"  - Arguments: {arguments}")
            self.logger.debug(f"  - Timeout: {timeout}s")
            
            tool_start = time.time()
            self.logger.info(f"[L.U.N.A. InteractionService] 도구 호출 시작: {server_id}/{tool_name}")

            coro = self.mcp_tool_manager.call_tool(server_id, tool_name, arguments, timeout=timeout)
            result = self._event_loop.run_until_complete(coro)
            
            tool_elapsed = time.time() - tool_start
            if tool_elapsed > timeout * 0.8:
                self.logger.warning(f"[L.U.N.A. InteractionService] 도구 호출 지연 중: {tool_elapsed:.2f}s / {timeout}s (80% 초과)")
            else:
                self.logger.info(f"[L.U.N.A. InteractionService] 도구 호출 완료: {tool_elapsed:.2f}s")

            result_str = str(result)
            if isinstance(result, dict):
                import json
                result_str = json.dumps(result, ensure_ascii=False)
            
            log_result = result_str[:100] if len(result_str) > 100 else result_str
            self.logger.info(f"[L.U.N.A. InteractionService] 도구 호출 성공: {server_id}/{tool_name}")
            self.logger.info(f"[L.U.N.A. InteractionService] 반환값: {log_result}")
            return result_str
        except asyncio.TimeoutError:
            self.logger.error(f"[L.U.N.A. InteractionService] '{server_id}/{tool_name}' 도구 호출 타임아웃 ({timeout}초)")
            raise
        except Exception as e:
            self.logger.error(f"[L.U.N.A. InteractionService] 도구 호출 실패: {server_id}/{tool_name}: {e}", exc_info=True)
            raise
    
    def _extract_tool_calls_from_text(self, text: str) -> tuple[list, int]:
        """
        LLM 텍스트 응답에서 도구 호출 추출
        형식: 'call:server_id/tool_name{...}'
        
        Returns:
            (tool_calls, last_tool_end_index) - 도구 호출 목록과 마지막 도구 호출의 끝 위치
        """
        tool_calls = []
        last_tool_end = 0
        
        # 패턴: 'call:server_id/tool_name{...}'
        pattern = r"call:([a-zA-Z0-9_/:.\-]+)"
        matches = list(re.finditer(pattern, text))
        
        for i, match in enumerate(matches):
            tool_id = match.group(1)
            
            start_idx = match.end()
            text_after = text[start_idx:].lstrip()
            
            if text_after.startswith('{'):
                brace_count = 0
                json_end = 0
                for j, char in enumerate(text_after):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = j + 1
                            break
                
                if json_end > 0:
                    args_str = text_after[:json_end]
                    try:
                        args = json.loads(args_str)
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"[L.U.N.A. MCP] JSON 파싱 실패: {args_str[:100]} ({e})")
                        args = {}
                    
                    tool_calls.append({
                        "id": f"tool_call_{i}",
                        "function": {
                            "name": tool_id,
                            "arguments": args
                        }
                    })
                    
                    lstrip_count = len(text[start_idx:]) - len(text_after)
                    last_tool_end = start_idx + lstrip_count + json_end
        
        return tool_calls, last_tool_end

    def _normalize(self, s: str) -> str:
        import re
        return re.sub(r'[^a-z0-9]', '', s.lower())

    # 폴백용 함수
    def _detect_and_suggest_tool(self, user_input: str, mcp_tools: list) -> dict | None:
        user_lower = user_input.lower()
        
        play_keywords = ["틀어", "재생", "플레이", "play", "켜", "켜줘", "들을래", "들으면서"]
        music_context = ["유튜브", "음악", "뮤직", "곡", "노래", "singer", "artist"]
        
        is_play_request = any(kw in user_lower for kw in play_keywords)
        is_music_context = any(kw in user_lower for kw in music_context)
        
        if is_play_request and is_music_context:
            track_name = self._extract_song_name(user_input)
            if track_name:
                for tool in mcp_tools:
                    tool_name = tool.get('function', {}).get('name', '').lower()
                    if 'playtrack' in tool_name or 'play' in tool_name:
                        self.logger.info(f"[L.U.N.A. InteractionService] 폴백: 음악 재생 감지")
                        return {
                            "id": "auto_tool_call_1",
                            "function": {
                                "name": tool.get('function', {}).get('name', ''),
                                "arguments": {"trackName": track_name}
                            }
                        }
        
        pause_keywords = ["멈춰", "멈추", "정지", "중지", "pause", "stop", "꺼", "끄"]
        if any(kw in user_lower for kw in pause_keywords) and is_music_context:
            for tool in mcp_tools:
                tool_name = tool.get('function', {}).get('name', '').lower()
                if 'pausetrack' in tool_name or 'pause' in tool_name:
                    self.logger.info(f"[L.U.N.A. InteractionService] 폴백: 음악 일시정지 감지")
                    return {
                        "id": "auto_tool_call_2",
                        "function": {
                            "name": tool.get('function', {}).get('name', ''),
                            "arguments": {}
                        }
                    }
        
        return None
    
    def _extract_song_name(self, text: str) -> str:
        import re
        
        match = re.search(r'([a-zA-Z0-9가-힣\s\-&]+?)\s*(?:틀어|재생|플레이|play|켜|들을래|listen)', text)
        if match:
            song = match.group(1).strip()
            if song:
                return song
        
        match = re.search(r'(?:에서|뮤직에서)\s+([a-zA-Z0-9가-힣\s\-&]+)(?:틀어|재생|플레이)', text)
        if match:
            song = match.group(1).strip()
            if song:
                return song
        
        words = text.split()
        if len(words) > 2:
            result = []
            for word in reversed(words):
                if any(kw in word.lower() for kw in ["틀어", "재생", "플레이", "play", "켜"]):
                    break
                result.insert(0, word)
            
            if result:
                return " ".join(result).strip()
        
        return text

    def _resolve_server_and_tool(self, raw_name: str, tools: list = None) -> tuple[str, str]:
        if "/" in raw_name:
            s, t = raw_name.split("/", 1)
            return s.strip(), t.strip()

        if tools is None:
            try:
                tools = self._get_available_mcp_tools() or []
            except Exception:
                tools = []
        else:
            tools = tools or []

        raw_norm = self._normalize(raw_name)
        candidates: list[tuple[str, str]] = []

        for it in tools:
            func = it.get("function", {}) if isinstance(it, dict) else {}
            tool_name = str(func.get("name", "")).strip()
            server_id = str(it.get("server_id") or it.get("server") or "").strip()
            if not server_id or not tool_name:
                continue

            pair = (server_id, tool_name)
            sv_norm = self._normalize(server_id)
            tl_norm = self._normalize(tool_name)
            cat_norm = self._normalize(server_id + tool_name)

            if raw_norm == cat_norm:
                candidates.append(pair)
                continue
            if raw_norm == tl_norm:
                candidates.append(pair)
                continue
            if raw_norm.startswith(sv_norm) and raw_norm.endswith(tl_norm):
                candidates.append(pair)
                continue

        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            def score(p: tuple[str, str]) -> int:
                return len(self._normalize(p[0] + p[1]))
            candidates.sort(key=score, reverse=True)
            return candidates[0]

        raise ValueError(f"Cannot resolve tool uniquely from name '{raw_name}'. Expected 'server/tool'.")

    def run(self, ko_input_text: str, use_tools: bool = False) -> InteractResponse:
        self.logger.info(f"[L.U.N.A. InteractionService] 사용자 입력: {ko_input_text} (도구 사용: {use_tools})")
        
        if use_tools and self.mcp_tool_manager:
            mcp_tools = self._get_available_mcp_tools()
            if mcp_tools:
                self.logger.info(f"[L.U.N.A. InteractionService] 에이전트 모드 활성화 - MCP 도구 {len(mcp_tools)}개")
                return self._run_agent_mode(ko_input_text)
        
        self.logger.info("[L.U.N.A. InteractionService] 일반 모드 처리")
        return self._run_normal_mode(ko_input_text)
        
    def _run_agent_mode(self, ko_input_text: str) -> InteractResponse:
        import time
        import re
        pipeline_start = time.time()
        
        is_api_mode = self.llm_service.get_mode() == "api"

        if is_api_mode:
            en_input = ko_input_text
            self.logger.info("[L.U.N.A. InteractionService] API 모드: 번역 생략, 한국어 입력 그대로 사용")
        else:
            try:
                en_input = self.translator_service.translate(ko_input_text, "ko", "en")
            except Exception as e:
                self.logger.warning(f"[L.U.N.A. InteractionService] 번역 실패, 원문 사용: {e}")
                en_input = ko_input_text

        context_messages = self.memory_service.get_context_for_llm()
        
        messages = []
        
        messages.extend([m for m in context_messages if m.get("role") != "system"])
        
        messages.append({"role": "user", "content": en_input})
        
        mcp_tools = self._get_available_mcp_tools()

        self.logger.info(f"[L.U.N.A. InteractionService] 사용 가능한 MCP 도구: {len(mcp_tools)}개")
        self.logger.info(f"[L.U.N.A. InteractionService] 대화 맥락: {len(context_messages)}개 메시지 포함")
        self.logger.debug(f"[L.U.N.A. InteractionService] 도구 목록: {[t.get('function', {}).get('name', 'unknown') for t in mcp_tools]}")

        llm_response = self.llm_service.generate(
            target=self.llm_target,
            system_prompt=self.rp_prompt_template,
            messages=messages,
            tools=mcp_tools if mcp_tools else None,
            skip_cache=True
        )
        if not llm_response or "choices" not in llm_response:
            return self.error_response("LLM 응답 처리 중 오류가 발생했습니다.")

        message = llm_response["choices"][0]["message"]
        response_content = message.get("content", "") or ""
        self.logger.info(f"[L.U.N.A. InteractionService] LLM 응답 content: {response_content[:100] if response_content else '(비어있음)'}...")
        tool_calls, tool_end_idx = self._extract_tool_calls_from_text(response_content)
        if not tool_calls:
            tool_calls = message.get("tool_calls") or []
        self.logger.info(f"[L.U.N.A. InteractionService] 도구 호출 발견: {len(tool_calls)}개")

        if not tool_calls:
            auto_tool_call = self._detect_and_suggest_tool(ko_input_text, mcp_tools)
            if auto_tool_call:
                self.logger.info(f"[L.U.N.A. InteractionService] 자동 도구 감지: {auto_tool_call['function']['name']}")
                tool_calls = [auto_tool_call]

        if not tool_calls:
            cleaned = response_content or ""

            cleaned = re.sub(r'\[CONTEXT\].*', '', cleaned, flags=re.IGNORECASE | re.DOTALL)

            cleaned = re.sub(
                r'✓\s*[^\n]*실행 결과[\s\S]*?```[\s\S]*?```',
                '',
                cleaned,
                flags=re.DOTALL
            )

            cleaned = re.sub(
                r'^\s*```[\s\S]*?```\s*',
                '',
                cleaned,
                flags=re.DOTALL
            ).strip()

            if not cleaned:
                cleaned = "그냥 네 생각 기다리는 중."

            response_content = cleaned

            final_ko = response_content if is_api_mode else self.translator_service.translate(response_content, "en", "ko")
            
            try:
                self.memory_service.add_entry(
                    user_input=ko_input_text,
                    assistant_response=final_ko,
                    metadata={
                        "mode": "agent",
                        "tool_called": False,
                        "tools_used": []
                    }
                )
                self.logger.info(f"[L.U.N.A. InteractionService] 대화 메모리 저장 완료 (도구 없음)")
            except Exception as e:
                self.logger.warning(f"[L.U.N.A. InteractionService] 메모리 저장 실패: {e}")
            
            top_emotion = "neutral"
            style, style_weight = get_style_from_emotion(top_emotion)
            ja = self.translator_service.translate(final_ko, "ko", "ja")
            tts = self.tts_service.synthesize(text=ja, style=style, style_weight=style_weight)
            
            pipeline_elapsed = time.time() - pipeline_start
            if pipeline_elapsed > 30:
                self.logger.warning(f"[L.U.N.A. InteractionService] 전체 처리 시간: {pipeline_elapsed:.2f}s (30초 초과 - 타임아웃 위험)")
            else:
                self.logger.info(f"[L.U.N.A. InteractionService] 전체 처리 완료: {pipeline_elapsed:.2f}s")
            
            return InteractResponse(text=final_ko, emotion=top_emotion, intent="agent", style=style, audio_url=tts.get("audio_url",""))

        tool_call = tool_calls[0]
        tool_name = tool_call["function"]["name"]
        raw_args = tool_call["function"].get("arguments", {})
        if isinstance(raw_args, str):
            try:
                import json
                tool_args = json.loads(raw_args)
            except Exception:
                self.logger.warning("[L.U.N.A. InteractionService] arguments JSON 파싱 실패 → {} 사용")
                tool_args = {}
        else:
            tool_args = raw_args if isinstance(raw_args, dict) else {}

        try:
            server_id, mcp_tool_name = self._resolve_server_and_tool(tool_name, mcp_tools)
        except Exception as e:
            ack_ko = f"도구 이름을 해석하지 못했어: {tool_name} ({e})"
            top_emotion = "neutral"
            style, style_weight = get_style_from_emotion(top_emotion)
            ja = self.translator_service.translate(ack_ko, "ko", "ja")
            tts = self.tts_service.synthesize(text=ja, style=style, style_weight=style_weight)
            return InteractResponse(text=ack_ko, emotion=top_emotion, intent="agent", style=style, audio_url=tts.get("audio_url",""))

        if not response_content:
            response_content = "알겠어."
            self.logger.info(f"[L.U.N.A. InteractionService] 빈 응답 → 기본 메시지 사용")
        
        if tool_end_idx > 0 and tool_end_idx < len(response_content):
            response_without_tool_call = response_content[tool_end_idx:].strip()
            self.logger.info(f"[L.U.N.A. InteractionService] 도구 호출 텍스트 제거됨 (위치: {tool_end_idx})")
        else:
            response_without_tool_call = response_content
        
        import re
        response_without_tool_call = re.sub(r'^[\}\s,]+', '', response_without_tool_call).strip()
        
        final_response = response_without_tool_call if response_without_tool_call else response_content
        
        final_ko = final_response if is_api_mode else self.translator_service.translate(final_response, "en", "ko")
        top_emotion = "neutral"
        style, style_weight = get_style_from_emotion(top_emotion)
        ja = self.translator_service.translate(final_ko, "ko", "ja")
        
        import time
        tts_start = time.time()
        self.logger.info(f"[L.U.N.A. InteractionService] 📍 음성 합성 시작")
        
        tts = self.tts_service.synthesize(text=ja, style=style, style_weight=style_weight)
        
        tts_elapsed = time.time() - tts_start
        if tts_elapsed > 15:
            self.logger.warning(f"[L.U.N.A. InteractionService] 음성 합성 지연 중: {tts_elapsed:.2f}s (15초 초과)")
        else:
            self.logger.info(f"[L.U.N.A. InteractionService] 음성 합성 완료: {tts_elapsed:.2f}s")
        
        try:
            self.memory_service.add_entry(
                user_input=ko_input_text,
                assistant_response=final_ko,
                metadata={
                    "mode": "agent",
                    "tool_called": len(tool_calls) > 0,
                    "tools_used": [tc["function"]["name"] for tc in tool_calls] if tool_calls else [],
                    "tool_pending_execution": True
                }
            )
            self.logger.info(f"[L.U.N.A. InteractionService] 대화 메모리 저장 완료 (도구 실행 대기 중)")
        except Exception as e:
            self.logger.warning(f"[L.U.N.A. InteractionService] 메모리 저장 실패: {e}")
        
        import asyncio # ignored
        import threading
        import time
        
        tool_result_buffer = {"result": None, "error": None}
        
        def run_tool_in_background():
            try:
                self.logger.info(f"[L.U.N.A. InteractionService] 백그라운드 도구 실행 시작: {server_id}/{mcp_tool_name}")
                self.logger.info(f"[L.U.N.A. InteractionService] 도구 인수: {tool_args}")
                result = self._call_mcp_tool(server_id, mcp_tool_name, tool_args)
                self.logger.info(f"[L.U.N.A. InteractionService] 백그라운드 도구 실행 완료!")
                self.logger.info(f"[L.U.N.A. InteractionService] 도구 반환값: {result[:200] if isinstance(result, str) else result}")
                
                tool_result_buffer["result"] = result
                
                try:
                    import json
                    
                    extracted_info = None
                    try:
                        result_text = result if isinstance(result, str) else str(result)

                        json_candidates = []
                        buf = []
                        depth = 0
                        for ch in result_text:
                            if ch == '{':
                                depth += 1
                            if depth > 0:
                                buf.append(ch)
                            if ch == '}':
                                depth -= 1
                                if depth == 0 and buf:
                                    cand = "".join(buf)
                                    buf = []
                                    try:
                                        obj = json.loads(cand)
                                        if isinstance(obj, dict):
                                            json_candidates.append(obj)
                                    except Exception:
                                        pass

                        target = None

                        if json_candidates:
                            candidates_with_id = [o for o in json_candidates if "id" in o]
                            if candidates_with_id:
                                target = max(candidates_with_id, key=lambda o: len(o.keys()))

                            if target is None:
                                status_like = [o for o in json_candidates if any(k in o for k in ("status", "code", "message"))]
                                if status_like:
                                    target = max(status_like, key=lambda o: len(o.keys()))

                            if target is None:
                                target = max(json_candidates, key=lambda o: len(o.keys()))

                        if target is not None:
                            extracted_info = {
                                "id": target.get("id", "N/A"),
                                "object": target.get("object", "N/A"),
                                "status": target.get("status", "N/A"),
                                "code": target.get("code", "N/A"),
                                "message": target.get("message", "N/A"),
                                "created_time": (
                                    target.get("created_time")
                                    or target.get("createdTime")
                                    or "N/A"
                                ),
                                "data": target,
                            }
                        else:
                            extracted_info = {"raw": result_text[:300]}

                    except Exception as parse_err:
                        self.logger.debug(f"[L.U.N.A. InteractionService] JSON 파싱 실패: {parse_err}")
                        extracted_info = {"raw": (result if isinstance(result, str) else str(result))[:300]}
                    
                    tool_result_message = (
                        f"{server_id}/{mcp_tool_name} 실행 결과\n"
                        f"{json.dumps(extracted_info, ensure_ascii=False, indent=2)}"
                    )

                    self.memory_service.add_entry(
                        user_input=f"[{mcp_tool_name} 도구 실행]",
                        assistant_response=tool_result_message,
                        metadata={
                            "mode": "tool_result",
                            "tool_name": mcp_tool_name,
                            "server_id": server_id,
                            "is_tool_result": True,
                            "tool_result_info": extracted_info,
                        }
                    )
                    self.logger.info(f"[L.U.N.A. InteractionService] 도구 결과를 메모리에 저장 완료: {extracted_info}")
                except Exception as me:
                    self.logger.warning(f"[L.U.N.A. InteractionService] 도구 결과 메모리 저장 실패: {me}")
                    
            except Exception as e:
                self.logger.error(f"[L.U.N.A. InteractionService] 백그라운드 도구 실행 오류: {e}", exc_info=True)
                tool_result_buffer["error"] = str(e)
            finally:
                time.sleep(0.5)
        
        tool_thread = threading.Thread(target=run_tool_in_background, daemon=False)
        tool_thread.start()
        
        self.logger.info(f"[L.U.N.A. InteractionService] 즉시 응답 반환 (도구는 백그라운드 실행)")
        
        pipeline_elapsed = time.time() - pipeline_start
        if pipeline_elapsed > 30:
            self.logger.warning(f"[L.U.N.A. InteractionService] 전체 처리 시간: {pipeline_elapsed:.2f}s (30초 초과 - 타임아웃 위험)")
        else:
            self.logger.info(f"[L.U.N.A. InteractionService] 전체 처리 완료: {pipeline_elapsed:.2f}s")
        
        return InteractResponse(text=final_ko, emotion=top_emotion, intent="agent", style=style, audio_url=tts.get("audio_url",""))
        
    def _run_normal_mode(self, ko_input_text: str) -> InteractResponse:
        try:
            is_api_mode = self.llm_service.get_mode() == "api"
            
            if is_api_mode:
                input_text = ko_input_text
                self.logger.info(f"[L.U.N.A. InteractionService] 한국어 입력 사용: {input_text}")
            else:
                input_text = self.translator_service.translate(ko_input_text, "ko", "en")
                self.logger.info(f"[L.U.N.A. InteractionService] 영어로 번역: {input_text}")
            
            emotion_probs = self.emotion_service.predict(input_text)
            top_emotion = max(emotion_probs, key=emotion_probs.get) if emotion_probs else "neutral"

            context_messages = self.memory_service.get_context_for_llm()
            
            messages = [
                {"role": "system", "content": self.rp_prompt_template}
            ]
            
            messages.extend(context_messages)
            
            messages.append({"role": "user", "content": input_text})

            llm_response_dict = self.llm_service.generate(
                target=self.llm_target,
                messages=messages
            )
            response_text = llm_response_dict["choices"][0]["message"]["content"].strip()
            self.logger.info(f"LLM 응답: {response_text}")

            if is_api_mode:
                ko_response = response_text
                self.logger.info(f"[L.U.N.A. InteractionService] 한국어 응답 사용: {ko_response}")
            else:
                ko_response = self.translator_service.translate(response_text, "en", "ko")
                self.logger.info(f"[L.U.N.A. InteractionService] 한국어로 번역: {ko_response}")
            
            self.memory_service.add_entry(
                user_input=ko_input_text,
                assistant_response=ko_response,
                metadata={
                    "emotion": top_emotion,
                    "intent": "general",
                    "mode": "api" if is_api_mode else "server"
                }
            )
            self.logger.info(f"[L.U.N.A. InteractionService] 대화 저장 완료")
            
            ja_text_for_tts = self.translator_service.translate(ko_response, "ko", "ja")
            style, style_weight = get_style_from_emotion(top_emotion)
            
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
