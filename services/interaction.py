# ====================================================================
#  File: services/interaction.py
# ====================================================================
import json
import logging
import asyncio
import re
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
        logger: logging.Logger = None,
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
        self.logger = logger or logging.getLogger("LUNA.Interaction")

        self._mcp_tools_cache: List[Dict[str, Any]] | None = None
        self._mcp_tools_cache_time: float = 0.0

        try:
            prompt_path = Path(prompt_dir)
            if llm_service.get_mode() == "api":
                self.rp_prompt_template = (prompt_path / "prompt_Luna_kr.txt").read_text(
                    encoding="utf-8"
                )
                print(
                    "[L.U.N.A. InteractionService] API 모드: 한국어 프롬프트(prompt_Luna_kr.txt) 로딩 완료."
                )
            else:
                self.rp_prompt_template = (prompt_path / "prompt_Kurumi.txt").read_text(
                    encoding="utf-8"
                )
                print(
                    "[L.U.N.A. InteractionService] 로컬 서버 모드: 영문 프롬프트(prompt_Kurumi.txt) 로딩 완료."
                )

            self.trans_prompt_template = (prompt_path / "prompt_Translate.txt").read_text(
                encoding="utf-8"
            )
        except Exception as e:
            print(f"[L.U.N.A. InteractionService] 프롬프트 로딩 실패: {e}")
            self.rp_prompt_template = "User: {user_input}\nAssistant:"
            self.trans_prompt_template = "Translate the following text to Korean."


    # ------------------------------------------------------------------
    # MCP / 도구 관련
    # ------------------------------------------------------------------
    def _get_available_mcp_tools(self) -> List[Dict[str, Any]]:
        """MCP 도구 목록 조회"""
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
            if self.logger:
                self.logger.warning(f"[L.U.N.A. InteractionService] 도구 목록 조회 실패: {e}")
            return []

    def _call_builtin_tool(self, tool_name: str, arguments: dict) -> dict | None:
        """내장 도구 처리"""

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
                "is_dst": bool(now.dst()),
            }

        return None

    def _call_mcp_tool(
        self,
        server_id: str,
        tool_name: str,
        arguments: dict,
        timeout: float | None = None,
    ) -> Any:
        """
        MCP 도구 호출 래퍼.
        - 내장 도구 먼저 확인
        - 그다음 MCPToolManager 통해 실제 호출
        """
        from mcp.types import CallToolResult, TextContent

        builtin_result = self._call_builtin_tool(f"{server_id}/{tool_name}", arguments)
        if builtin_result is not None:
            if self.logger:
                self.logger.info(
                    f"[L.U.N.A. InteractionService] 내장 도구 사용: {server_id}/{tool_name}"
                )
            import json

            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps(builtin_result, ensure_ascii=False),
                    )
                ],
                isError=False,
            )

        if not self.mcp_tool_manager:
            raise RuntimeError("MCP 도구 매니저가 없습니다.")

        if timeout is None:
            timeout = float(os.getenv("LUNA_TOOL_TIMEOUT", "15"))

        try:
            import time

            if not hasattr(self, "_event_loop"):
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_closed():
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    self._event_loop = loop
                except RuntimeError:
                    self._event_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(self._event_loop)

            if self.logger:
                self.logger.debug("[L.U.N.A. InteractionService] 도구 호출 상세 정보:")
                self.logger.debug(f"  - Server: {server_id}")
                self.logger.debug(f"  - Tool: {tool_name}")
                self.logger.debug(f"  - Arguments: {arguments}")
                self.logger.debug(f"  - Timeout: {timeout}s")

            tool_start = time.time()
            if self.logger:
                self.logger.info(
                    f"[L.U.N.A. InteractionService] 도구 호출 시작: {server_id}/{tool_name}"
                )

            coro = self.mcp_tool_manager.call_tool(
                server_id, tool_name, arguments, timeout=timeout
            )
            result = self._event_loop.run_until_complete(coro)

            tool_elapsed = time.time() - tool_start
            if self.logger:
                if tool_elapsed > timeout * 0.8:
                    self.logger.warning(
                        f"[L.U.N.A. InteractionService] 도구 호출 지연 중: {tool_elapsed:.2f}s / {timeout}s (80% 초과)"
                    )
                else:
                    self.logger.info(
                        f"[L.U.N.A. InteractionService] 도구 호출 완료: {tool_elapsed:.2f}s"
                    )

                log_result = str(result)[:100]
                self.logger.info(
                    f"[L.U.N.A. InteractionService] 도구 호출 성공: {server_id}/{tool_name}"
                )
                self.logger.info(f"[L.U.N.A. InteractionService] 반환값: {log_result}")

            return result

        except asyncio.TimeoutError:
            if self.logger:
                self.logger.error(
                    f"[L.U.N.A. InteractionService] '{server_id}/{tool_name}' 도구 호출 타임아웃 ({timeout}초)"
                )
            raise
        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"[L.U.N.A. InteractionService] 도구 호출 실패: {server_id}/{tool_name}: {e}",
                    exc_info=True,
                )
            raise
    
    # ------------------------------------------------------------------
    # LLM 출력에서 도구 호출 파싱 / 도구 이름 해석
    # ------------------------------------------------------------------
    def _extract_tool_calls_from_text(self, text: str) -> Tuple[List[Dict[str, Any]], int]:
        """
        LLM 텍스트 응답에서 도구 호출 추출
        형식: 'call:server_id/tool_name{...}'
        """
        tool_calls: List[Dict[str, Any]] = []
        last_tool_end = 0

        pattern = r"call:([a-zA-Z0-9_/:.\-]+)"
        matches = list(re.finditer(pattern, text))

        for i, match in enumerate(matches):
            tool_id = match.group(1)

            start_idx = match.end()
            text_after = text[start_idx:].lstrip()

            if text_after.startswith("{"):
                brace_count = 0
                json_end = 0
                for j, char in enumerate(text_after):
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = j + 1
                            break

                if json_end > 0:
                    args_str = text_after[:json_end]
                    try:
                        args = json.loads(args_str)
                    except json.JSONDecodeError as e:
                        if self.logger:
                            self.logger.warning(
                                f"[L.U.N.A. MCP] JSON 파싱 실패: {args_str[:100]} ({e})"
                            )
                        args = {}

                    tool_calls.append(
                        {
                            "id": f"tool_call_{i}",
                            "function": {
                                "name": tool_id,
                                "arguments": args,
                            },
                        }
                    )

                    lstrip_count = len(text[start_idx:]) - len(text_after)
                    last_tool_end = start_idx + lstrip_count + json_end

        return tool_calls, last_tool_end

    def _normalize(self, s: str) -> str:
        import re as _re

        return _re.sub(r"[^a-z0-9]", "", s.lower())

    def _detect_and_suggest_tool(
        self,
        user_input: str,
        mcp_tools: list,
        llm_had_content: bool = False,
        last_assistant: str | None = None,
    ) -> dict | None:
        user_lower = user_input.lower()
        last_assistant = last_assistant or ""
        last_lower = last_assistant.lower()

        if llm_had_content:
            self.logger.info("[L.U.N.A. InteractionService] LLM 응답 존재 → 폴백 로직 스킵")
            return None

        import re
        
        ack_patterns = [
            r"^ㅇㅇ$", r"^응$", r"^그래$", r"^맞아$", r"^좋아$",
            r"^ㅇㅋ$", r"^응응$", r"^웅$"
        ]
        is_ack = any(re.search(pat, user_lower.strip()) for pat in ack_patterns)


        time_question_patterns = [
            r"몇\s*시",
            r"지금\s*(시간|몇시)",
            r"현재\s*시간",
            r"what\s*time",
        ]

        is_time_request = any(re.search(pattern, user_lower) for pattern in time_question_patterns)

        non_time_contexts = ["시간이", "시간을", "시간에", "시간나면", "시간있", "시간동안"]
        if any(ctx in user_lower for ctx in non_time_contexts):
            is_time_request = False

        if is_time_request:
            for tool in mcp_tools:
                func = tool.get("function", {}) or {}
                tool_name = (func.get("name") or "").lower()
                if "time" in tool_name and ("get" in tool_name or "current" in tool_name):
                    self.logger.info("[L.U.N.A. InteractionService] 폴백: 시간 질문 감지 → 강제 도구 호출")
                    return {
                        "id": "auto_tool_call_time",
                        "function": {
                            "name": func.get("name", ""),
                            "arguments": {}
                        }
                    }

        weather_keywords = [
            "날씨", "weather", "기온", "온도",
            "비와", "비 와", "눈와", "눈 와", "맑아", "흐려"
        ]
        is_weather_request = any(kw in user_lower for kw in weather_keywords)

        if is_weather_request:
            for tool in mcp_tools:
                func = tool.get("function", {}) or {}
                tool_name = (func.get("name") or "").lower()
                if "weather" in tool_name:
                    self.logger.info("[L.U.N.A. InteractionService] 폴백: 날씨 질문 감지 → 강제 도구 호출")
                    return {
                        "id": "auto_tool_call_weather",
                        "function": {
                            "name": func.get("name", ""),
                            "arguments": {"location": "Seoul"}
                        }
                    }
                    
        # --------------------
        # Notion: 새 페이지 생성 감지
        # --------------------
        notion_keywords = ["노션", "notion", "데이터베이스"]
        page_keywords = ["페이지", "page"]
        create_keywords = ["만들", "생성", "추가"]

        # 현재 턴 + 직전 어시스턴트 둘 다 본다
        is_notion_ctx = any(k in user_lower for k in notion_keywords) or any(
            k in last_lower for k in notion_keywords
        )
        is_page_ctx = any(k in user_lower for k in page_keywords) or any(
            k in last_lower for k in page_keywords
        )
        is_create_ctx = any(k in user_lower for k in create_keywords) or any(
            k in last_lower for k in create_keywords
        )
        
        if is_ack and is_notion_ctx and is_page_ctx and is_create_ctx:
            self.logger.info("[L.U.N.A. InteractionService] 폴백: Notion DB 안에 새 페이지 생성 확인 응답 감지")

            notion_tools = [
                t for t in mcp_tools
                if "notion" in str(t.get("server_id", "")).lower()
                or "notion" in str(t.get("function", {}).get("name", "")).lower()
            ]

            target_tool = None
            for tool in notion_tools:
                fname = str(tool.get("function", {}).get("name", "")).lower()
                if "post-page" in fname or "create-a-page" in fname or "post_page" in fname:
                    target_tool = tool
                    break

            if target_tool:
                func_name = target_tool["function"]["name"]

                db_id = None
                m = re.search(r"([0-9a-f]{32})", last_assistant)
                if not m:
                    m = re.search(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", last_assistant)
                if m:
                    db_id = m.group(1).replace("-", "")

                args = {}
                if db_id:
                    args["parent"] = {"database_id": db_id}

                args.setdefault("properties", {
                    "title": [{"text": {"content": "루나"}}]
                })
                args.setdefault("children", [{
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{
                            "type": "text",
                            "text": {"content": "다엘을 위해 존재하는 루나. 언제나 다엘 곁에 있을게."}
                        }]
                    }
                }])

                self.logger.info(f"[L.U.N.A. InteractionService] 자동 도구 감지: {func_name} (Notion DB 페이지 생성)")
                return {
                    "id": "auto_tool_call_notion_db_page",
                    "function": {
                        "name": func_name,
                        "arguments": args,
                    },
                }

        search_keywords = [
            "검색", "찾아봐", "찾아봐줘", "찾아줘",
            "알아봐", "알아봐줘", "알아봐 줄래",
            "search", "서치", "서쳐줘", "서치해줘",

            "추천해줘", "추천 해줘", "추천해줄래", "추천해 줄래",
            "추천좀", "추천 좀", "추천 부탁",
            "추천할만한", "추천할 만한",
            "뭐가 좋아", "뭐가 맛있", "뭐 먹을까"
        ]
        is_search_request = any(kw in user_lower for kw in search_keywords)

        if is_search_request:
            ddg_search_tool = None
            for tool in mcp_tools:
                sid = (tool.get("server_id") or tool.get("server") or "").lower()
                func = tool.get("function", {}) or {}
                fname = (func.get("name") or "").lower()

                if sid == "ddg-search" and ("search" in fname or fname == "search"):
                    ddg_search_tool = tool
                    break

            target_tool = None

            if ddg_search_tool:
                target_tool = ddg_search_tool
                self.logger.info("[L.U.N.A. InteractionService] 폴백: ddg-search 전용 검색 도구 선택")
            else:
                best_tool = None
                best_score = -999

                for tool in mcp_tools:
                    func = tool.get("function", {}) or {}
                    fname = (func.get("name") or "").lower()
                    desc = (tool.get("description") or "").lower()
                    sid = (tool.get("server_id") or tool.get("server") or "").lower()

                    score = 0
                    if "search" in fname:
                        score += 5
                    if "search" in desc or "search" in sid:
                        score += 3
                    if any(x in fname or x in desc for x in ["web", "browser", "internet"]):
                        score += 2
                    if "fetch_content" in fname or "fetch" in fname:
                        score -= 2

                    if score > best_score:
                        best_score = score
                        best_tool = tool

                if best_tool and best_score > 0:
                    target_tool = best_tool

            if target_tool:
                func = target_tool.get("function", {}) or {}
                schema = func.get("parameters") or target_tool.get("inputSchema") or {}
                props = (schema.get("properties") or {}) if isinstance(schema, dict) else {}

                candidate_keys = ["query", "q", "search", "text", "prompt", "input", "keyword", "keywords"]
                arg_key = None
                for k in candidate_keys:
                    if k in props:
                        arg_key = k
                        break

                if not arg_key and len(props) == 1:
                    arg_key = next(iter(props.keys()), None)

                args = {arg_key: user_input} if arg_key else {}

                self.logger.info(
                    f"[L.U.N.A. InteractionService] 폴백: 검색/추천 요청 감지 → "
                    f"'{func.get('name', '')}' 호출 예정 (arg_key={arg_key})"
                )
                return {
                    "id": "auto_tool_call_search",
                    "function": {
                        "name": func.get("name", ""),
                        "arguments": args
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
                    func = tool.get("function", {}) or {}
                    tool_name = (func.get("name") or "").lower()
                    if "playtrack" in tool_name or "play" in tool_name:
                        self.logger.info("[L.U.N.A. InteractionService] 폴백: 음악 재생 감지")
                        return {
                            "id": "auto_tool_call_music_play",
                            "function": {
                                "name": func.get("name", ""),
                                "arguments": {"trackName": track_name}
                            }
                        }

        pause_keywords = ["멈춰", "멈추", "정지", "중지", "pause", "stop", "꺼", "끄"]
        if any(kw in user_lower for kw in pause_keywords) and is_music_context:
            for tool in mcp_tools:
                func = tool.get("function", {}) or {}
                tool_name = (func.get("name") or "").lower()
                if "pausetrack" in tool_name or "pause" in tool_name:
                    self.logger.info("[L.U.N.A. InteractionService] 폴백: 음악 일시정지 감지")
                    return {
                        "id": "auto_tool_call_music_pause",
                        "function": {
                            "name": func.get("name", ""),
                            "arguments": {}
                        }
                    }

        return None
    
    def _extract_song_name(self, text: str) -> str:
        import re

        match = re.search(
            r"([a-zA-Z0-9가-힣\s\-&]+?)\s*(?:틀어|재생|플레이|play|켜|들을래|listen)", text
        )
        if match:
            song = match.group(1).strip()
            if song:
                return song

        match = re.search(
            r"(?:에서|뮤직에서)\s+([a-zA-Z0-9가-힣\s\-&]+)(?:틀어|재생|플레이)", text
        )
        if match:
            song = match.group(1).strip()
            if song:
                return song

        words = text.split()
        if len(words) > 2:
            result: List[str] = []
            for word in reversed(words):
                if any(
                    kw in word.lower()
                    for kw in ["틀어", "재생", "플레이", "play", "켜"]
                ):
                    break
                result.insert(0, word)

            if result:
                return " ".join(result).strip()

        return text
    
    def _extract_notion_title(self, text: str) -> str:
        import re
        m = re.search(r"(.+?)라는\s*제목", text)
        if m:
            return m.group(1).strip()

        m = re.search(r"제목은\s+(.+)", text)
        if m:
            return m.group(1).strip()

        return text.strip()[:30]

    def _extract_notion_body(self, text: str) -> str:
        import re
        body = re.sub(r".+?라는\s*제목(으로|에)?", "", text).strip()
        if body:
            return body
        return ""

    def _resolve_server_and_tool(
        self,
        raw_name: str,
        tools: List[Dict[str, Any]] | None = None,
    ) -> Tuple[str, str]:
        """
        Gemini가 준 function.name(예: 'playTrack')을
        실제 MCP server/tool 조합으로 매핑.
        """
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
        candidates: List[Tuple[str, str]] = []

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
            def score(p: Tuple[str, str]) -> int:
                return len(self._normalize(p[0] + p[1]))

            candidates.sort(key=score, reverse=True)
            return candidates[0]

        raise ValueError(
            f"Cannot resolve tool uniquely from name '{raw_name}'. "
            "Expected 'server/tool'."
        )
        
    # ------------------------------------------------------------------
    # 감정 및 의도 분석
    # ------------------------------------------------------------------
    def _analyze_emotion_and_intent(
        self,
        ko_text: str,
    ) -> Tuple[str, str, Dict[str, float], Dict[str, float]]:
        """
        한국어 텍스트(ko_text)에 대해
        - EmotionService / MultiIntentService를 한 번씩만 호출해서
        - 최고 감정 레이블, 최고 의도 레이블, 전체 점수 딕셔너리를 반환한다.

        반환:
            model_emotion: 감정 분류 모델 기준 최고 감정 (예: 'joy', 'sadness', 'something_else')
            top_intent:    의도 분류 모델 기준 최고 의도 (예: 'greeting', 'weather' 등, 없으면 'general')
            emotion_probs: {감정 레이블: 점수}
            intent_probs:  {의도 레이블: 점수}
        """
        emotion_probs = self.emotion_service.predict(
            ko_text,
            skip_translation=True,
        )
        intent_probs = self.multi_intent_service.predict(
            ko_text,
            skip_translation=True,
        )

        if emotion_probs:
            model_emotion = max(emotion_probs, key=emotion_probs.get)
        else:
            model_emotion = "neutral"

        if intent_probs:
            top_intent = max(intent_probs, key=intent_probs.get)
        else:
            top_intent = "general"

        return model_emotion, top_intent, emotion_probs, intent_probs

        
    # ------------------------------------------------------------------
    # 엔트리 포인트
    # ------------------------------------------------------------------
    def run(
        self,
        ko_input_text: str,
        use_tools: bool = False,
        skip_tts_generation: bool = False,
    ) -> InteractResponse:
        if self.logger:
            self.logger.info(
                f"[L.U.N.A. InteractionService] 사용자 입력: {ko_input_text} (도구 사용: {use_tools})"
            )

        if use_tools and self.mcp_tool_manager:
            mcp_tools = self._get_available_mcp_tools()
            if mcp_tools:
                if self.logger:
                    self.logger.info(
                        f"[L.U.N.A. InteractionService] 에이전트 모드 활성화 - MCP 도구 {len(mcp_tools)}개"
                    )
                return self._run_agent_mode(
                    ko_input_text,
                    skip_tts_generation=skip_tts_generation,
                )

        if self.logger:
            self.logger.info("[L.U.N.A. InteractionService] 일반 모드 처리")
        return self._run_normal_mode(
            ko_input_text,
            skip_tts_generation=skip_tts_generation,
        )
        
    # ------------------------------------------------------------------
    # 에이전트 모드 (도구 사용)
    # ------------------------------------------------------------------
    def _run_agent_mode(
        self,
        ko_input_text: str,
        skip_tts_generation: bool = False,
    ) -> InteractResponse:
        import time
        pipeline_start = time.time()
        
        is_api_mode = self.llm_service.get_mode() == "api"

        if is_api_mode:
            user_text = ko_input_text
            if self.logger:
                self.logger.info(
                    "[L.U.N.A. InteractionService] API 모드: 번역 생략, 한국어 입력 그대로 사용"
                )
        else:
            try:
                user_text = self.translator_service.translate(ko_input_text, "ko", "en")
            except Exception as e:
                if self.logger:
                    self.logger.warning(
                        f"[L.U.N.A. InteractionService] 번역 실패, 원문 사용: {e}"
                    )
                user_text = ko_input_text
                
        # ------------------------------------------------------------------
        # 감정 / 의도 분석
        # ------------------------------------------------------------------
        top_emotion, top_intent, emotion_probs, intent_probs = self._analyze_emotion_and_intent(user_text)
        
        if self.logger:
            emo_str = ", ".join([f"{k}:{v:.2f}" for k, v in emotion_probs.items()])
            intent_str = ", ".join([f"{k}:{v:.2f}" for k, v in intent_probs.items()])
            self.logger.info(
                f"[L.U.N.A. InteractionService] 감정 분석 결과(모델): {top_emotion} ({emo_str})"
            )
            self.logger.info(
                f"[L.U.N.A. InteractionService] 의도 분석 결과: {top_intent} ({intent_str})"
            )
            
        # ------------------------------------------------------------------
        # LLM 호출 (도구 스키마 포함)
        # ------------------------------------------------------------------
        context_messages = self.memory_service.get_full_context_for_llm()

        messages: List[Dict[str, Any]] = list(context_messages)
        messages.append({"role": "user", "content": user_text})

        mcp_tools = self._get_available_mcp_tools()
        
        if self.logger:
            self.logger.info(
                f"[L.U.N.A. InteractionService] 사용 가능한 MCP 도구: {len(mcp_tools)}개"
            )
            self.logger.info(
                f"[L.U.N.A. InteractionService] 대화 맥락: {len(context_messages)}개 메시지 포함"
            )
            self.logger.debug(
                "[L.U.N.A. InteractionService] 도구 목록: "
                + str(
                    [t.get("function", {}).get("name", "unknown") for t in mcp_tools]
                )
            )

        llm_response = self.llm_service.generate(
            target=self.llm_target,
            system_prompt=self.rp_prompt_template,
            messages=messages,
            tools=mcp_tools if mcp_tools else None,
            skip_cache=True,
        )
        if not llm_response or "choices" not in llm_response:
            return self.error_response("LLM 응답 처리 중 오류가 발생했습니다.")

        message = llm_response["choices"][0]["message"]
        response_content = message.get("content", "") or ""
        if self.logger:
            self.logger.info(
                "[L.U.N.A. InteractionService] LLM 응답 content: "
                f"{response_content[:100] if response_content else '(비어있음)'}..."
            )
            
        last_assistant_text = ""
        for m in reversed(context_messages):
            if m.get("role") == "assistant":
                last_assistant_text = m.get("content", "") or ""
                break

        tool_calls, _ = self._extract_tool_calls_from_text(response_content)

        if not tool_calls:
            tool_calls = message.get("tool_calls") or []

        if self.logger:
            self.logger.info(
                f"[L.U.N.A. InteractionService] 도구 호출 발견: {len(tool_calls)}개"
            )

        if not tool_calls:
            llm_had_meaningful_response = bool(response_content and len(response_content.strip()) > 10)
            auto_tool_call = self._detect_and_suggest_tool(
                ko_input_text, 
                mcp_tools,
                llm_had_content=llm_had_meaningful_response,
                last_assistant=last_assistant_text
            )
            if auto_tool_call:
                if self.logger:
                    self.logger.info(
                        "[L.U.N.A. InteractionService] 자동 도구 감지: "
                        f"{auto_tool_call['function']['name']}"
                    )
                tool_calls = [auto_tool_call]

        final_ko = ""
        tool_used_flag = False

        if not tool_calls:
            cleaned = response_content or ""

            cleaned = re.sub(r'\[CONTEXT\].*', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
            cleaned = re.sub(r'✓\s*[^\n]*실행 결과[\s\S]*?```[\s\S]*?```', '', cleaned, flags=re.DOTALL)
            cleaned = re.sub(r'^\s*```[\s\S]*?```\s*', '', cleaned, flags=re.DOTALL).strip()
            cleaned = re.sub(r'\[THOUGHT\][\s\S]*?(\[\/THOUGHT\]|$)', '', cleaned, flags=re.IGNORECASE).strip()
            cleaned = re.sub(
                r'^(생각|思考|thinking|thought)\s*[:：]\s*.*?(?=\n\n|\n[^a-zA-Z가-힣]|$)',
                '', cleaned, flags=re.IGNORECASE | re.DOTALL
            ).strip()
            cleaned = re.sub(r'\[\/?CHARACTER\]', '', cleaned, flags=re.IGNORECASE).strip()
            cleaned = re.sub(r'^call:[^\n]+\n?', '', cleaned, flags=re.IGNORECASE | re.MULTILINE).strip()

            if not cleaned:
                cleaned = "조금만 더 구체적으로 말해줄래, 다엘?"

            final_ko = cleaned if is_api_mode else self.translator_service.translate(cleaned, "en", "ko")
            tool_used_flag = False

        # ------------------------------------------------------------------
        # 도구 사용이 필요한 경우
        # ------------------------------------------------------------------
        else:
            tool_used_flag = True
            tool_call = tool_calls[0]
            tool_name_raw = tool_call["function"]["name"]
            raw_args = tool_call["function"].get("arguments", {})

            if isinstance(raw_args, str):
                try:
                    tool_args = json.loads(raw_args)
                except Exception:
                    if self.logger:
                        self.logger.warning(
                            "[L.U.N.A. InteractionService] arguments JSON 파싱 실패 → {} 사용"
                        )
                    tool_args = {}
            else:
                tool_args = raw_args if isinstance(raw_args, dict) else {}

            try:
                server_id, mcp_tool_name = self._resolve_server_and_tool(
                    tool_name_raw,
                    mcp_tools,
                )
            except Exception as e:
                final_ko = f"도구 이름을 해석하지 못했어: {tool_name_raw} ({e})"
                server_id, mcp_tool_name = "unknown", tool_name_raw

            if not final_ko:
                if self.logger:
                    self.logger.info(
                        f"[L.U.N.A. InteractionService] 도구 실행 시작: {server_id}/{mcp_tool_name}"
                    )
                    self.logger.info(
                        f"[L.U.N.A. InteractionService] 도구 인수: {tool_args}"
                    )

                try:
                    tool_result = self._call_mcp_tool(
                        server_id,
                        mcp_tool_name,
                        tool_args,
                    )
                    if self.logger:
                        self.logger.info(
                            "[L.U.N.A. InteractionService] 도구 실행 완료!"
                        )
                        self.logger.info(
                            "[L.U.N.A. InteractionService] 도구 반환값: "
                            f"{str(tool_result)[:300]}"
                        )

                    result_text = ""
                    extracted = False

                    try:
                        if hasattr(tool_result, "content") and tool_result.content:
                            for content_item in tool_result.content:
                                if (
                                    hasattr(content_item, "text")
                                    and content_item.text
                                ):
                                    result_text = content_item.text
                                    extracted = True
                                    break
                    except Exception as e:
                        if self.logger:
                            self.logger.warning(
                                f"[L.U.N.A. InteractionService] content 직접 접근 실패: {e}"
                            )

                    if not extracted or not result_text:
                        tool_result_str = str(tool_result)
                        text_start = tool_result_str.find("text='")
                        if text_start != -1:
                            text_start += 6
                            end_marker = tool_result_str.find(
                                "', annotations", text_start
                            )
                            if end_marker == -1:
                                end_marker = tool_result_str.find("')", text_start)
                            if end_marker == -1:
                                end_marker = tool_result_str.find("']", text_start)

                            if end_marker != -1:
                                result_text = tool_result_str[text_start:end_marker]
                                result_text = (
                                    result_text.replace("\\n", "\n")
                                    .replace("\\'", "'")
                                )

                        if not result_text:
                            json_start = tool_result_str.find("{")
                            if json_start != -1:
                                depth = 0
                                json_end = -1
                                for i, ch in enumerate(
                                    tool_result_str[json_start:], json_start
                                ):
                                    if ch == "{":
                                        depth += 1
                                    elif ch == "}":
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

                    if self.logger:
                        self.logger.info(
                            "[L.U.N.A. InteractionService] 추출된 텍스트: "
                            f"{result_text[:200]}"
                        )

                    # 메모리에 저장할 요약 데이터
                    extracted_data = None
                    try:
                        json_candidates = []
                        buf: List[str] = []
                        depth = 0
                        for ch in result_text:
                            if ch == "{":
                                depth += 1
                            if depth > 0:
                                buf.append(ch)
                            if ch == "}":
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
                            extracted_data = max(
                                json_candidates,
                                key=lambda o: len(o.keys()),
                            )
                    except Exception:
                        pass

                    if self.logger:
                        self.logger.info(
                            "[L.U.N.A. InteractionService] 도구 결과를 바탕으로 LLM 후속 응답 생성 중..."
                        )

                    followup_messages: List[Dict[str, Any]] = list(messages)
                    followup_messages.append(
                        {
                            "role": "assistant",
                            "content": response_content,
                        }
                    )

                    safe_tool_result = result_text[:1000] if result_text else "Success"

                    followup_messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"[System: 도구 '{mcp_tool_name}' 실행 결과]\n"
                                f"{safe_tool_result}\n\n"
                                "이 결과를 바탕으로 사용자에게 자연스럽게 대답해줘. "
                                "(도구/시스템 언급 없이, 한국어로 1~2문장)"
                            ),
                        }
                    )

                    final_llm_response = self.llm_service.generate(
                        target=self.llm_target,
                        system_prompt=self.rp_prompt_template,
                        messages=followup_messages,
                        tools=None,
                        skip_cache=True
                    )
                    
                    if final_llm_response and "choices" in final_llm_response:
                        final_ko = final_llm_response["choices"][0]["message"]["content"].strip()
                        self.logger.info(f"[L.U.N.A. InteractionService] LLM 후속 응답(raw): {final_ko}")

                        final_ko = re.sub(
                            r'^call:[^\n]+\n?', '',
                            final_ko,
                            flags=re.IGNORECASE | re.MULTILINE
                        ).strip()
                        final_ko = re.sub(r'^\s*```[\s\S]*?```\s*', '', final_ko, flags=re.DOTALL).strip()
                        final_ko = re.sub(r'\[\/?CHARACTER\]', '', final_ko, flags=re.IGNORECASE).strip()
                        self.logger.info(f"[L.U.N.A. InteractionService] LLM 후속 응답(clean): {final_ko}")
                    else:
                        final_ko = "작업을 완료했어, 다엘."

                    try:
                        tool_result_for_memory: Any = (
                            extracted_data
                            if extracted_data is not None
                            else {"raw": result_text[:300]}
                        )

                        self.memory_service.add_entry(
                            user_input=ko_input_text,
                            assistant_response=final_ko,
                            metadata={
                                "mode": "agent",
                                "tool_called": True,
                                "tool_name": mcp_tool_name,
                                "server_id": server_id,
                                "tool_result": tool_result_for_memory,
                                "emotion": top_emotion,
                                "intent": top_intent,
                                "emotion_probs": emotion_probs,
                                "intent_probs": intent_probs,
                            },
                        )
                        if self.logger:
                            self.logger.info(
                                "[L.U.N.A. InteractionService] 도구 결과 메모리 저장 완료"
                            )
                    except Exception as me:
                        if self.logger:
                            self.logger.warning(
                                f"[L.U.N.A. InteractionService] 메모리 저장 실패: {me}"
                            )

                except Exception as e:
                    if self.logger:
                        self.logger.error(
                            f"[L.U.N.A. InteractionService] 도구 실행 오류: {e}",
                            exc_info=True,
                        )
                    final_ko = "도구 실행 중에 문제가 생겼어, 다엘."

        # ------------------------------------------------------------------
        # 감정 추정 (Unity 감정 맵핑)
        # ------------------------------------------------------------------
        unity_emotion = "Neutral"
        try:
            if "사랑" in final_ko or "좋아" in final_ko:
                unity_emotion = "yandere1"
            elif "부끄" in final_ko or "쑥쓰" in final_ko:
                unity_emotion = "shy1"
            elif "미안" in final_ko or "슬퍼" in final_ko:
                unity_emotion = "sad1"
            elif "화나" in final_ko:
                unity_emotion = "anger1"
            elif "재밌" in final_ko or "기뻐" in final_ko:
                unity_emotion = "smile1"

            if self.logger:
                self.logger.info(
                    f"[L.U.N.A. Analysis] 보낼 표정(에이전트 모드): {unity_emotion}"
                )
        except Exception:
            pass

        style, style_weight = get_style_from_emotion(top_emotion)

        # ------------------------------------------------------------------
        # TTS
        # ------------------------------------------------------------------
        audio_url = ""
        if not skip_tts_generation:
            try:
                ja = self.translator_service.translate(final_ko, "ko", "ja")
                tts_start = time.time()
                if self.logger:
                    self.logger.info("[L.U.N.A. InteractionService] 📍 음성 합성 시작")

                tts = self.tts_service.synthesize(
                    text=ja,
                    style=style,
                    style_weight=style_weight,
                )
                audio_url = tts.get("audio_url", "")

                tts_elapsed = time.time() - tts_start
                if self.logger:
                    if tts_elapsed > 15:
                        self.logger.warning(
                            f"[L.U.N.A. InteractionService] 음성 합성 지연: {tts_elapsed:.2f}s"
                        )
                    else:
                        self.logger.info(
                            f"[L.U.N.A. InteractionService] 음성 합성 완료: {tts_elapsed:.2f}s"
                        )
            except Exception as e:
                if self.logger:
                    self.logger.error(f"TTS 생성 실패: {e}")
        else:
            if self.logger:
                self.logger.info("[L.U.N.A. InteractionService] TTS 스킵")

        if not tool_used_flag:
            try:
                self.memory_service.add_entry(
                    user_input=ko_input_text,
                    assistant_response=final_ko,
                    metadata={
                        "mode": "agent",
                        "tool_called": False,
                        "tools_used": [],
                        "emotion": unity_emotion,
                        "emotion_model": top_emotion,
                        "intent": top_intent,
                        "emotion_probs": emotion_probs,
                        "intent_probs": intent_probs,
                    },
                )
            except Exception as e:
                if self.logger:
                    self.logger.warning(
                        f"[L.U.N.A. InteractionService] 메모리 저장 실패: {e}"
                    )

        pipeline_elapsed = time.time() - pipeline_start
        if self.logger:
            self.logger.info(
                f"[L.U.N.A. InteractionService] 전체 처리 완료: {pipeline_elapsed:.2f}s"
            )

        # TTS 제외 → audio_url은 항상 빈 문자열
        return InteractResponse(
            text=final_ko,
            emotion=unity_emotion,
            intent=top_intent,
            style=style,
            audio_url="",
        )
        
    # ------------------------------------------------------------------
    # 일반 모드 (도구 없음)
    # ------------------------------------------------------------------
    def _run_normal_mode(
        self,
        ko_input_text: str,
        skip_tts_generation: bool = False,
    ) -> InteractResponse:
        try:
            is_api_mode = self.llm_service.get_mode() == "api"

            if is_api_mode:
                en_input = ko_input_text
                if self.logger:
                    self.logger.info(
                        f"[L.U.N.A] 일반모드 - 한국어 입력 그대로 사용: {en_input}"
                    )
            else:
                en_input = self.translator_service.translate(
                    ko_input_text, "ko", "en"
                )
                if self.logger:
                    self.logger.info(
                        f"[L.U.N.A] 일반모드 - 입력 영어 번역: {en_input}"
                    )

            top_emotion, top_intent, emotion_probs, intent_probs = self._analyze_emotion_and_intent(en_input)

            if self.logger:
                self.logger.info(
                    f"[L.U.N.A] 감정/의도 분석 결과 - emotion: {top_emotion}, intent: {top_intent}"
                )

            context_messages = self.memory_service.get_full_context_for_llm()

            llm_resp = self.llm_service.generate(
                target=self.llm_target,
                system_prompt=self.rp_prompt_template,
                messages=context_messages + [
                    {"role": "user", "content": en_input}
                ]
            )

            en_response = llm_resp["choices"][0]["message"]["content"].strip()
            if self.logger:
                self.logger.info(f"[L.U.N.A] LLM 응답(영문): {en_response}")

            if is_api_mode:
                ko_response = en_response
            else:
                ko_response = self.translator_service.translate(
                    en_response, "en", "ko"
                )

            if self.logger:
                self.logger.info(
                    f"[L.U.N.A] 한국어 응답: {ko_response}"
                )

            self.memory_service.add_entry(
                user_input=ko_input_text,
                assistant_response=ko_response,
                metadata={
                    "emotion": top_emotion,
                    "intent": top_intent,
                    "mode": "api" if is_api_mode else "server",
                    "emotion_probs": emotion_probs,
                    "intent_probs": intent_probs,
                },
            )
            
            unity_emotion = "Neutral"
            try:
                text = final_ko or ""

                if "사랑해" in text or "사랑해요" in text or "너를 사랑해" in text or "다엘을 사랑해" in text:
                    unity_emotion = "yandere1"
                elif "좋아해" in text and "다엘" in text:
                    unity_emotion = "yandere1"
                elif "부끄" in text or "쑥쓰" in text:
                    unity_emotion = "shy1"
                elif "미안" in text or "슬퍼" in text or "ㅠㅠ" in text or "ㅜㅜ" in text:
                    unity_emotion = "sad1"
                elif "화나" in text or "짜증" in text or "열받" in text:
                    unity_emotion = "anger1"
                elif "재밌" in text or "기뻐" in text or "즐거워" in text or "신나" in text:
                    unity_emotion = "smile1"

                if self.logger:
                    self.logger.info(
                        f"[L.U.N.A. Analysis] 보낼 표정(에이전트 모드): {unity_emotion}"
                    )
            except Exception:
                unity_emotion = "Neutral"

            style, style_weight = get_style_from_emotion(top_emotion)

            audio_url = ""
            if not skip_tts_generation:
                ja_text = self.translator_service.translate(
                    ko_response, "ko", "ja"
                )

                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop = asyncio.get_event_loop()

                tts_result = loop.run_until_complete(
                    self.tts_service.synthesize_async(
                        text=ja_text,
                        style=style,
                        style_weight=style_weight,
                    )
                )
                audio_url = tts_result.get("audio_url", "")
                
            self.memory_service.add_entry(
                user_input=ko_input_text,
                assistant_response=ko_response,
                metadata={
                    "emotion": unity_emotion,
                    "emotion_model": top_emotion,
                    "intent": top_intent,
                    "mode": "api" if is_api_mode else "server",
                    "emotion_probs": emotion_probs,
                    "intent_probs": intent_probs,
                },
            )
            if self.logger:
                self.logger.info("[L.U.N.A. InteractionService] 대화 저장 완료 (일반 모드)")


            return InteractResponse(
                text=ko_response,
                emotion=unity_emotion,
                intent=top_intent,
                style=style,
                audio_url=audio_url,
            )

        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"[L.U.N.A] 일반 모드 처리 중 오류: {e}",
                    exc_info=True
                )
            return self.error_response("일반 응답 처리 중 오류가 발생했습니다.")

    # ------------------------------------------------------------------
    # 오류 응답
    # ------------------------------------------------------------------
    def error_response(self, error_message: str) -> InteractResponse:
        if self.logger:
            self.logger.error(f"오류 응답 반환: {error_message}")
        return InteractResponse(
            text=f"지금은 조금 문제가 있어. 다시 시도해볼래, 다엘?",
            emotion="neutral",
            intent="error",
            style="Neutral",
            audio_url="",
        )