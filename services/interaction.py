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
    
    def _call_builtin_tool(self, tool_name: str, arguments: dict) -> dict | None:
        """내장 도구 처리 (MCP 없이 빠르게 실행)"""
        
        # 시간 도구 - MCP 대신 직접 처리
        if "time" in tool_name.lower() and "get_current_time" in tool_name:
            from datetime import datetime
            import pytz
            
            tz_name = arguments.get("timezone", "Asia/Seoul")
            try:
                tz = pytz.timezone(tz_name)
            except Exception:
                tz = pytz.timezone("Asia/Seoul")
            
            now = datetime.now(tz)
            return {
                "timezone": tz_name,
                "datetime": now.isoformat(),
                "day_of_week": now.strftime("%A"),
                "is_dst": bool(now.dst())
            }
        
        return None  # 내장 도구 없음 → MCP로 처리
    
    def _call_mcp_tool(self, server_id: str, tool_name: str, arguments: dict, timeout: float | None = None) -> str:
        # 내장 도구 먼저 확인 (MCP보다 훨씬 빠름)
        builtin_result = self._call_builtin_tool(f"{server_id}/{tool_name}", arguments)
        if builtin_result is not None:
            self.logger.info(f"[L.U.N.A. InteractionService] 내장 도구 사용: {server_id}/{tool_name}")
            # MCP CallToolResult 형식으로 래핑
            from mcp.types import CallToolResult, TextContent
            import json
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(builtin_result, ensure_ascii=False))],
                isError=False
            )
        
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

            # 로그용으로만 문자열 변환
            log_result = str(result)[:100]
            self.logger.info(f"[L.U.N.A. InteractionService] 도구 호출 성공: {server_id}/{tool_name}")
            self.logger.info(f"[L.U.N.A. InteractionService] 반환값: {log_result}")
            
            # 원본 객체 반환 (TextContent 등 접근 가능하도록)
            return result
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
    def _detect_and_suggest_tool(self, user_input: str, mcp_tools: list, llm_had_content: bool = False) -> dict | None:
        user_lower = user_input.lower()
        
        if llm_had_content:
            self.logger.info(f"[L.U.N.A. InteractionService] LLM 응답 존재 → 폴백 로직 스킵")
            return None
        
        import re
        time_question_patterns = [
            r"몇\s*시",  # "몇시", "몇 시"
            r"지금\s*(시간|몇시)",  # "지금 시간", "지금 몇시"
            r"현재\s*시간",  # "현재 시간"
            r"what\s*time",  # "what time"
        ]
    
        is_time_request = any(re.search(pattern, user_lower) for pattern in time_question_patterns)
    
        non_time_contexts = ["시간이", "시간을", "시간에", "시간나면", "시간있", "시간동안"]
        if any(ctx in user_lower for ctx in non_time_contexts):
            is_time_request = False
        
        if is_time_request:
            for tool in mcp_tools:
                tool_name = tool.get('function', {}).get('name', '').lower()
                if 'time' in tool_name and ('get' in tool_name or 'current' in tool_name):
                    self.logger.info(f"[L.U.N.A. InteractionService] 폴백: 시간 질문 감지 → 강제 도구 호출")
                    return {
                        "id": "auto_tool_call_time",
                        "function": {
                            "name": tool.get('function', {}).get('name', ''),
                            "arguments": {}
                        }
                    }
        
        # 날씨 질문 감지
        weather_keywords = ["날씨", "weather", "기온", "온도", "비와", "비 와", "눈와", "눈 와", "맑아", "흐려"]
        is_weather_request = any(kw in user_lower for kw in weather_keywords)
        
        if is_weather_request:
            for tool in mcp_tools:
                tool_name = tool.get('function', {}).get('name', '').lower()
                if 'weather' in tool_name:
                    self.logger.info(f"[L.U.N.A. InteractionService] 폴백: 날씨 질문 감지 → 강제 도구 호출")
                    return {
                        "id": "auto_tool_call_weather",
                        "function": {
                            "name": tool.get('function', {}).get('name', ''),
                            "arguments": {"location": "Seoul"}
                        }
                    }
        
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

    def run(self, ko_input_text: str, use_tools: bool = False, skip_tts_generation: bool = False) -> InteractResponse:
        self.logger.info(f"[L.U.N.A. InteractionService] 사용자 입력: {ko_input_text} (도구 사용: {use_tools})")
        
        if use_tools and self.mcp_tool_manager:
            mcp_tools = self._get_available_mcp_tools()
            if mcp_tools:
                self.logger.info(f"[L.U.N.A. InteractionService] 에이전트 모드 활성화 - MCP 도구 {len(mcp_tools)}개")
                return self._run_agent_mode(ko_input_text, skip_tts_generation=skip_tts_generation)
        
        self.logger.info("[L.U.N.A. InteractionService] 일반 모드 처리")
        return self._run_normal_mode(ko_input_text, skip_tts_generation=skip_tts_generation)
        
    def _run_agent_mode(self, ko_input_text: str, skip_tts_generation: bool = False) -> InteractResponse:
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

        context_messages = self.memory_service.get_full_context_for_llm()
        
        # 이전 방식
        # messages = []
        # messages.extend([m for m in context_messages if m.get("role") != "system"])
        # messages.append({"role": "user", "content": en_input})
        
        # 시스템 프롬프트 포함 방식
        messages = context_messages.copy()
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
            llm_had_meaningful_response = bool(response_content and len(response_content.strip()) > 10)
            auto_tool_call = self._detect_and_suggest_tool(
                ko_input_text, 
                mcp_tools,
                llm_had_content=llm_had_meaningful_response
            )
            if auto_tool_call:
                self.logger.info(f"[L.U.N.A. InteractionService] 자동 도구 감지: {auto_tool_call['function']['name']}")
                tool_calls = [auto_tool_call]
                
        final_ko = ""
        tool_used_flag = False

        if not tool_calls:
            cleaned = response_content or ""

            cleaned = re.sub(r'\[CONTEXT\].*', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
            
            cleaned = re.sub(r'✓\s*[^\n]*실행 결과[\s\S]*?```[\s\S]*?```', '', cleaned, flags=re.DOTALL)
            
            cleaned = re.sub(r'^\s*```[\s\S]*?```\s*', '', cleaned, flags=re.DOTALL).strip()
            
            cleaned = re.sub(r'\[THOUGHT\][\s\S]*?(\[\/THOUGHT\]|$)', '', cleaned, flags=re.IGNORECASE).strip()
            
            cleaned = re.sub(r'^(생각|思考|thinking|thought)\s*[:：]\s*.*?(?=\n\n|\n[^a-zA-Z가-힣]|$)', '', cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
            
            cleaned = re.sub(r'\[\/?CHARACTER\]', '', cleaned, flags=re.IGNORECASE).strip()

            if not cleaned: 
                cleaned = "음... (생각 중)"

            final_ko = cleaned if is_api_mode else self.translator_service.translate(cleaned, "en", "ko")
            tool_used_flag = False
            
        else:
            tool_used_flag = True
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
                final_ko = f"도구 이름을 해석하지 못했어: {tool_name} ({e})"
                server_id, mcp_tool_name = "unknown", tool_name
                
            if not final_ko:
                self.logger.info(f"[L.U.N.A. InteractionService] 도구 실행 시작: {server_id}/{mcp_tool_name}")
                self.logger.info(f"[L.U.N.A. InteractionService] 도구 인수: {tool_args}")
                
                try:
                    import json
                    tool_result = self._call_mcp_tool(server_id, mcp_tool_name, tool_args)
                    self.logger.info(f"[L.U.N.A. InteractionService] 도구 실행 완료!")
                    self.logger.info(f"[L.U.N.A. InteractionService] 도구 반환값: {str(tool_result)[:300]}")
                    
                    result_text = ""
                    if isinstance(tool_result, str):
                        result_text = tool_result
                    else:
                        extracted = False
                        
                        try:
                            if hasattr(tool_result, 'content') and tool_result.content:
                                for content_item in tool_result.content:
                                    if hasattr(content_item, 'text') and content_item.text:
                                        result_text = content_item.text
                                        extracted = True
                                        break
                        except Exception as e:
                            self.logger.warning(f"[L.U.N.A. InteractionService] content 직접 접근 실패: {e}")
                        
                        if not extracted or not result_text:
                            tool_result_str = str(tool_result)
                            text_start = tool_result_str.find("text='")
                            if text_start != -1:
                                text_start += 6
                                end_marker = tool_result_str.find("', annotations", text_start)
                                if end_marker == -1:
                                    end_marker = tool_result_str.find("')", text_start)
                                if end_marker == -1:
                                    end_marker = tool_result_str.find("']", text_start)
                                
                                if end_marker != -1:
                                    result_text = tool_result_str[text_start:end_marker]
                                    result_text = result_text.replace('\\n', '\n').replace("\\'", "'")
                            
                            if not result_text:
                                json_start = tool_result_str.find('{')
                                if json_start != -1:
                                    depth = 0
                                    json_end = -1
                                    for i, ch in enumerate(tool_result_str[json_start:], json_start):
                                        if ch == '{':
                                            depth += 1
                                        elif ch == '}':
                                            depth -= 1
                                            if depth == 0:
                                                json_end = i + 1
                                                break
                                    if json_end > json_start:
                                        result_text = tool_result_str[json_start:json_end]
                                    else:
                                        result_text = tool_result_str
                                else:
                                    result_text = tool_result_str
                    
                    self.logger.info(f"[L.U.N.A. InteractionService] 추출된 텍스트: {result_text[:200]}")
                    
                    extracted_data = None
                    try:
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
                        
                        if json_candidates:
                            extracted_data = max(json_candidates, key=lambda o: len(o.keys()))
                    except Exception:
                        pass
                    
                    self.logger.info(f"[L.U.N.A. InteractionService] 도구 결과를 바탕으로 LLM 후속 응답 생성 중...")
                    
                    followup_messages = messages.copy()
                    
                    followup_messages.append({
                        "role": "assistant",
                        "content": response_content
                    })
                    
                    safe_tool_result = result_text[:1000] if result_text else "Success"
                    
                    followup_messages.append({
                        "role": "user",
                        "content": f"[System: 도구 '{mcp_tool_name}' 실행 결과입니다]\n{safe_tool_result}\n\n이 결과를 바탕으로 사용자에게 자연스럽게 대답해줘. (일본어가 아닌 한국어로)"
                    })
                    
                    final_llm_response = self.llm_service.generate(
                        target=self.llm_target,
                        system_prompt=self.rp_prompt_template,
                        messages=followup_messages,
                        tools=None,
                        skip_cache=True
                    )
                    
                    if final_llm_response and "choices" in final_llm_response:
                        final_ko = final_llm_response["choices"][0]["message"]["content"].strip()
                        self.logger.info(f"[L.U.N.A. InteractionService] LLM 후속 응답: {final_ko}")
                    else:
                        final_ko = "작업을 완료했어, 다엘."
                    
                    try:
                        tool_result_message = f"{server_id}/{mcp_tool_name} 실행 결과\n{result_text[:500]}"
                        self.memory_service.add_entry(
                            user_input=ko_input_text,
                            assistant_response=final_ko,
                            metadata={
                                "mode": "agent",
                                "tool_called": True,
                                "tool_name": mcp_tool_name,
                                "server_id": server_id,
                                "tool_result": extracted_data if extracted_data else {"raw": result_text[:300]},
                            }
                        )
                        self.logger.info(f"[L.U.N.A. InteractionService] 도구 결과 메모리 저장 완료")
                    except Exception as me:
                        self.logger.warning(f"[L.U.N.A. InteractionService] 메모리 저장 실패: {me}")
                        
                except Exception as e:
                    self.logger.error(f"[L.U.N.A. InteractionService] 도구 실행 오류: {e}", exc_info=True)
                    final_ko = "도구 실행 중 문제가 생겼어, 다엘."
                    
        top_emotion = "Neutral"
        
        try:
            if "사랑" in final_ko or "좋아" in final_ko: 
                top_emotion = "yandere1"
            elif "부끄" in final_ko or "쑥쓰" in final_ko: 
                top_emotion = "shy1"
            elif "미안" in final_ko or "슬퍼" in final_ko: 
                top_emotion = "sad1"
            elif "화나" in final_ko:
                top_emotion = "anger1"
            elif "재밌" in final_ko or "기뻐" in final_ko:
                top_emotion = "smile1"

            self.logger.info(f"[L.U.N.A. Analysis] 보낼 표정: {top_emotion}")
        except:
            pass

        style, style_weight = get_style_from_emotion(top_emotion)
        
        audio_url = ""
        if not skip_tts_generation:
            try:
                ja = self.translator_service.translate(final_ko, "ko", "ja")
                import time
                tts_start = time.time()
                self.logger.info(f"[L.U.N.A. InteractionService] 📍 음성 합성 시작")
                
                tts = self.tts_service.synthesize(text=ja, style=style, style_weight=style_weight)
                audio_url = tts.get("audio_url", "")
                
                tts_elapsed = time.time() - tts_start
                if tts_elapsed > 15:
                    self.logger.warning(f"[L.U.N.A. InteractionService] 음성 합성 지연: {tts_elapsed:.2f}s")
                else:
                    self.logger.info(f"[L.U.N.A. InteractionService] 음성 합성 완료: {tts_elapsed:.2f}s")
            except Exception as e:
                self.logger.error(f"TTS 생성 실패: {e}")
        else:
            self.logger.info(f"[L.U.N.A. InteractionService] TTS 스킵")

        if not tool_used_flag:
            try:
                self.memory_service.add_entry(
                    user_input=ko_input_text,
                    assistant_response=final_ko,
                    metadata={
                        "mode": "agent", 
                        "tool_called": False, 
                        "tools_used": [],
                        "emotion": top_emotion
                    }
                )
                self.logger.info(f"[L.U.N.A. InteractionService] 대화 메모리 저장 완료 (도구 없음)")
            except Exception as e:
                self.logger.warning(f"[L.U.N.A. InteractionService] 메모리 저장 실패: {e}")

        pipeline_elapsed = time.time() - pipeline_start
        self.logger.info(f"[L.U.N.A. InteractionService] 전체 처리 완료: {pipeline_elapsed:.2f}s")
        
        return InteractResponse(text=final_ko, emotion=top_emotion, intent="agent", style=style, audio_url=audio_url)
        
    def _run_normal_mode(self, ko_input_text: str, skip_tts_generation: bool = False) -> InteractResponse:
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

            context_messages = self.memory_service.get_full_context_for_llm()
            
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
                
            try:
                analysis_text = ko_response
                if any(ord(c) > 127 for c in ko_response):
                    try:
                        analysis_text = self.translator_service.translate(ko_response, "ko", "en")
                        self.logger.info(f"[L.U.N.A. Analysis] 감정 분석용 영어 번역: {analysis_text[:30]}...")
                    except:
                        pass
                    
                final_emotion_probs = self.emotion_service.predict(analysis_text)
                if final_emotion_probs:
                    top_emotion = max(final_emotion_probs, key=final_emotion_probs.get)
                
                if "사랑" in ko_response or "좋아해" in ko_response or "반짝" in ko_response: 
                    top_emotion = "love"
                elif "부끄" in ko_response or "쑥쓰" in ko_response or "헤헤" in ko_response: 
                    top_emotion = "shy"
                elif "미안" in ko_response or "슬퍼" in ko_response or "ㅠㅠ" in ko_response: 
                    top_emotion = "sadness"
                elif "화나" in ko_response or "바보" in ko_response:
                    top_emotion = "anger"
                elif "재밌" in ko_response or "ㅋㅋㅋ" in ko_response:
                    top_emotion = "joy"
                else:
                    pass
                    
                self.logger.info(f"[L.U.N.A. Analysis] 일반 모드 감정: {top_emotion}")
            except Exception:
                pass
            
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
            
            style, style_weight = get_style_from_emotion(top_emotion)
            
            audio_url = ""
            if not skip_tts_generation:
                ja_text_for_tts = self.translator_service.translate(ko_response, "ko", "ja")
                
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
                audio_url = tts_result.get("audio_url", "")

            return InteractResponse(
                text=ko_response, emotion=top_emotion, intent="general",
                style=style, audio_url=audio_url
            )
        except Exception as e:
            self.logger.error(f"파이프라인 실행 중 오류: {e}", exc_info=True)
            return self.error_response("일반 응답 처리 중 오류가 발생했습니다.")

    def error_response(self, error_message: str) -> InteractResponse:
        self.logger.error(f"오류 응답 반환: {error_message}")
        return InteractResponse(text=f"오류: {error_message}", emotion="neutral", intent="error", style="Neutral", audio_url="")
