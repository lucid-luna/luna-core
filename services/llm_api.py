# ====================================================================
#  File: services/llm_api.py
# ====================================================================

import os
import hashlib
from typing import Dict, List, Any, Optional, Tuple

from google import genai
from google.genai import types as gx
from dotenv import load_dotenv

from .cache import ResponseCache

load_dotenv()

class LLMAPIService:
    """
    L.U.N.A. 외부 LLM API 래퍼
    """
    def __init__(
        self,
        api_configs: Dict[str, Dict[str, str]],
        enable_cache: bool = True,
        cache_ttl: int = 3600
    ):
        self.api_configs = api_configs
        self.clients: Dict[str, genai.Client] = {}
        self.enable_cache = enable_cache
        
        if enable_cache:
            self.cache = ResponseCache(
                cache_dir="./cache",
                ttl=cache_ttl,
                max_cache_size=100,
                similarity_threshold=0.85
            )
            print("[L.U.N.A. LLM API] 응답 캐싱 활성화됨")
        else:
            self.cache = None
        
        self._initialize_clients()
    
    # ------------------------------------------------------------------
    # 클라이언트 초기화
    # ------------------------------------------------------------------
    def _initialize_clients(self) -> None:
        """API 클라이언트 초기화"""
        for provider, config in self.api_configs.items():
            if provider != "gemini":
                continue

            api_key = (
                config.get("api_key")
                or os.getenv("GEMINI_API_KEY")
                or os.getenv("GENAI_API_KEY")
            )
            if not api_key:
                print("[L.U.N.A. LLM API] Gemini API 키가 설정되지 않았습니다.")
                continue

            try:
                self.clients[provider] = genai.Client(api_key=api_key)
                print("[L.U.N.A. LLM API] Gemini 클라이언트 초기화 완료")
            except Exception as e:
                print(f"[L.U.N.A. LLM API] Gemini 클라이언트 초기화 실패: {e}")

    def _sanitize_tool_name_for_gemini(self, tool_name: str) -> str:
        if not tool_name:
            return "unknown_tool"

        import re

        sanitized = re.sub(r"[^a-zA-Z0-9_.:_\-]", "", tool_name)

        if sanitized and not (sanitized[0].isalpha() or sanitized[0] == "_"):
            sanitized = "_" + sanitized

        if not sanitized:
            sanitized = "unknown_tool"

        sanitized = sanitized[:64]
        return sanitized

    def _clean_schema_for_gemini(self, schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            if not isinstance(schema, dict):
                return schema

            ALLOWED = {
                "type",
                "description",
                "properties",
                "items",
                "required",
                "enum",
                "default",
                "format",
                "minimum",
                "maximum",
                "minLength",
                "maxLength",
                "pattern",
                "minItems",
                "maxItems",
                "anyOf",
                "oneOf",
                "allOf",
                "const",
            }
            DROP_EXACT = {
                "$defs",
                "$schema",
                "$id",
                "definitions",
                "additionalProperties",
                "additional_properties",
                "unevaluatedProperties",
                "unevaluated_properties",
                "readOnly",
                "writeOnly",
                "deprecated",
                "examples",
                "example",
                "title",
            }

            cleaned: Dict[str, Any] = {}

            for key, value in schema.items():
                if key in DROP_EXACT:
                    continue

                norm_key = key
                if key == "additional_properties":
                    continue
                elif key == "min_items":
                    norm_key = "minItems"
                elif key == "max_items":
                    norm_key = "maxItems"
                elif key == "min_length":
                    norm_key = "minLength"
                elif key == "max_length":
                    norm_key = "maxLength"

                if norm_key not in ALLOWED:
                    continue

                if norm_key == "type":
                    if isinstance(value, list):
                        cleaned[norm_key] = value[0] if value else "object"
                    else:
                        cleaned[norm_key] = value

                elif norm_key == "properties":
                    if isinstance(value, dict):
                        subprops: Dict[str, Any] = {}
                        for p_name, p_schema in value.items():
                            if isinstance(p_schema, dict):
                                sub_clean = self._clean_schema_for_gemini(p_schema)
                                if sub_clean:
                                    subprops[p_name] = sub_clean
                        cleaned[norm_key] = subprops

                elif norm_key == "items":
                    if isinstance(value, dict):
                        sub = self._clean_schema_for_gemini(value)
                        if sub:
                            cleaned[norm_key] = sub
                    elif isinstance(value, list) and value:
                        first = value[0]
                        if isinstance(first, dict):
                            sub = self._clean_schema_for_gemini(first)
                            if sub:
                                cleaned[norm_key] = sub

                elif norm_key in ("anyOf", "oneOf", "allOf"):
                    if isinstance(value, list):
                        arr = []
                        for v in value:
                            if isinstance(v, dict):
                                sub = self._clean_schema_for_gemini(v)
                                if sub:
                                    arr.append(sub)
                        if arr:
                            cleaned[norm_key] = arr

                elif norm_key == "required":
                    if isinstance(value, list):
                        props = cleaned.get("properties", {})
                        req = [r for r in value if isinstance(r, str) and (r in props)]
                        if req:
                            cleaned[norm_key] = req

                else:
                    cleaned[norm_key] = value

            if "type" not in cleaned and ("properties" in cleaned or "required" in cleaned):
                cleaned["type"] = "object"

            return cleaned or {"type": "object", "properties": {}}
        except Exception as e:
            print(f"[L.U.N.A. LLM API] 스키마 정제 중 오류: {e}")
            return {"type": "object", "properties": {}}

    def _to_contents(self, messages: Optional[List[Dict[str, Any]]], fallback_text: str) -> List[gx.Content]:
        if not messages:
            return [gx.Content(role="user", parts=[gx.Part(text=fallback_text)])]

        out: List[gx.Content] = []

        for m in messages:
            role = m.get("role", "user")

            if role == "tool":
                name = m.get("name") or m.get("tool_name") or "unknown_tool"
                raw = m.get("content", "")
                resp_obj = raw
                if isinstance(raw, str):
                    try:
                        import json

                        resp_obj = json.loads(raw)
                    except Exception:
                        resp_obj = {"_raw": raw}

                part = gx.Part(
                    function_response=gx.FunctionResponse(
                        name=name,
                        response=resp_obj,
                    )
                )
                out.append(gx.Content(role="user", parts=[part]))
                continue

            role_map = {"system": "user", "user": "user", "assistant": "model"}
            grole = role_map.get(role, "user")
            txt = str(m.get("content", ""))
            out.append(gx.Content(role=grole, parts=[gx.Part(text=txt)]))

        return out

    def _extract_gemini_output(self, resp) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Gemini 응답 객체에서:
        - 일반 텍스트
        - function_call 정보(Gemini 스타일)
        를 뽑아서 (text, tool_calls) 형태로 반환.
        """
        texts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []

        try:
            candidates = getattr(resp, "candidates", None) or []
            if not candidates:
                fallback_txt = getattr(resp, "text", None) or getattr(resp, "content", None)
                if isinstance(fallback_txt, str):
                    return fallback_txt.strip(), []
                return str(resp), []

            content = getattr(candidates[0], "content", None)
            parts = getattr(content, "parts", None) or []

            for part in parts:
                ptext = getattr(part, "text", None)
                if isinstance(ptext, str) and ptext:
                    texts.append(ptext)
                    continue

                if hasattr(part, "function_call"):
                    fc = part.function_call
                    args_obj = getattr(fc, "args", {}) or {}
                    if not isinstance(args_obj, dict):
                        try:
                            import json

                            args_obj = json.loads(str(args_obj))
                        except Exception:
                            args_obj = {"_raw": str(args_obj)}

                    tool_calls.append(
                        {
                            "name": getattr(fc, "name", "") or "",
                            "arguments": args_obj,
                        }
                    )

        except Exception:
            return str(resp), []

        return "\n".join(texts).strip(), tool_calls
    
    def generate(
        self,
        target_provider: str,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        skip_cache: bool = False,
    ) -> Dict[str, Any]:
        """
        LLM API 호출 엔트리 포인트
        """
        if target_provider not in self.api_configs:
            print(f"[L.U.N.A. LLM API] 알 수 없는 API 제공자입니다: {target_provider}")
            return {"error": "Unknown provider"}

        if target_provider not in self.clients:
            print(f"[L.U.N.A. LLM API] '{target_provider}' 클라이언트가 초기화되지 않았습니다.")
            return {"error": "Client not initialized"}

        config = self.api_configs[target_provider]

        if target_provider == "gemini":
            return self._generate_gemini(
                config=config,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                messages=messages,
                model=model,
                stream=stream,
                tools=tools,
                skip_cache=skip_cache,
            )

        return {"error": "Unsupported provider"}

    
    def _generate_gemini(
        self,
        config: Dict[str, str],
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        skip_cache: bool = False,
    ) -> Dict[str, Any]:
        client = self.clients["gemini"]
        model_name = model or config.get("model", "gemini-2.5-flash")

        if messages:
            contents = self._to_contents(
                messages,
                fallback_text=messages[-1].get("content", "") if messages else "",
            )
        else:
            if system_prompt is None or user_prompt is None:
                print("[L.U.N.A. LLM API] 시스템 프롬프트와 사용자 프롬프트는 필수입니다.")
                return {"error": "Missing prompts"}
            contents = self._to_contents(None, fallback_text=user_prompt)

        user_input = None
        if messages:
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    user_input = msg.get("content", "")
                    break
        elif user_prompt:
            user_input = user_prompt

        context_hash = ""
        if messages and len(messages) > 1:
            context_str = str(messages[:-1])
            context_hash = hashlib.md5(context_str.encode("utf-8")).hexdigest()[:8]

        has_tool_results = any(m.get("role") == "tool" for m in (messages or []))
        has_tools = bool(tools)

        if (
            self.cache
            and user_input
            and not skip_cache
            and not has_tools
            and not has_tool_results
        ):
            cached = self.cache.get(
                prompt=user_input,
                model=model_name,
                context_hash=context_hash,
                use_similarity=True,
            )
            if cached:
                return cached

        gen_config = gx.GenerateContentConfig()

        print(
            f"[L.U.N.A. LLM API] 디버그: tools={len(tools) if tools else 0}, "
            f"system_prompt={'있음' if system_prompt else '없음'}"
        )

        if system_prompt:
            gen_config.system_instruction = system_prompt

        if tools:
            function_declarations = []

            for tool in tools:
                func = tool.get("function", tool)

                input_schema = func.get("inputSchema", {}) or {}
                cleaned_schema = self._clean_schema_for_gemini(input_schema)

                function_declarations.append(
                    gx.FunctionDeclaration(
                        name=func.get("name", ""),
                        description=func.get("description", ""),
                        parameters=cleaned_schema,
                    )
                )

            gen_config.tools = [gx.Tool(function_declarations=function_declarations)]
            print(f"[L.U.N.A. LLM API] ✅ 도구 {len(tools)}개 스키마 주입 완료")
        else:
            print("[L.U.N.A. LLM API] 도구 없음, 스키마 주입 생략")

        api_kwargs: Dict[str, Any] = {
            "model": model_name,
            "contents": contents,
        }
        if getattr(gen_config, "system_instruction", None) or getattr(
            gen_config, "tools", None
        ):
            api_kwargs["config"] = gen_config

        try:
            if stream:
                return self._generate_gemini_stream(
                    client=client,
                    model_name=model_name,
                    contents=contents,
                    gen_config=gen_config if ("config" in api_kwargs) else None,
                    user_input=user_input,
                    context_hash=context_hash,
                )

            import time
            import logging

            logger = logging.getLogger("LUNA_API")
            llm_start = time.time()
            logger.info(f"[LLM] Gemini API 호출 시작: {model_name}")

            response = None
            retry_count = 0
            max_retries = 1

            while retry_count <= max_retries:
                try:
                    retry_start = time.time()

                    try:
                        response = client.models.generate_content(**api_kwargs)
                    except TypeError as te:
                        if "config" in str(te) or "unexpected keyword argument" in str(te):
                            logger.warning(
                                f"[L.U.N.A. LLM API] config 매개변수 오류, 제거 후 재시도: {te}"
                            )
                            api_kwargs.pop("config", None)
                            response = client.models.generate_content(**api_kwargs)
                        else:
                            raise

                    retry_elapsed = time.time() - retry_start
                    logger.info(
                        f"[L.U.N.A. LLM API] API 호출 성공 "
                        f"(시도 {retry_count + 1}/{max_retries + 1}, 소요: {retry_elapsed:.2f}s)"
                    )
                    break

                except (TimeoutError, ConnectionError, Exception) as e:
                    retry_elapsed = time.time() - retry_start
                    logger.warning(
                        f"[L.U.N.A. LLM API] API 호출 실패 "
                        f"(시도 {retry_count + 1}/{max_retries + 1}, 소요: {retry_elapsed:.2f}s): "
                        f"{type(e).__name__}: {str(e)[:100]}"
                    )

                    if retry_count < max_retries:
                        retry_count += 1
                        wait_time = 2 ** retry_count
                        logger.info(f"[L.U.N.A. LLM API] {wait_time}초 후 재시도...")
                        time.sleep(wait_time)
                    else:
                        raise

            llm_elapsed = time.time() - llm_start
            if llm_elapsed > 20:
                logger.warning(
                    f"[L.U.N.A. LLM API] LLM API 호출이 20초를 초과했습니다: {llm_elapsed:.2f}s (타임아웃 위험)"
                )
            else:
                logger.info(f"[L.U.N.A. LLM API] LLM API 호출 완료: {llm_elapsed:.2f}s")

            text, tool_calls = self._extract_gemini_output(response)

            import json
            from uuid import uuid4

            openai_tool_calls = []
            for tc in (tool_calls or []):
                try:
                    name = tc.get("name", "") or ""
                    args = tc.get("arguments", {}) or {}
                    if not isinstance(args, str):
                        args_str = json.dumps(args, ensure_ascii=False)
                    else:
                        args_str = args
                    openai_tool_calls.append(
                        {
                            "id": f"call_{uuid4().hex[:8]}",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": args_str,
                            },
                        }
                    )
                except Exception:
                    openai_tool_calls.append(
                        {
                            "id": f"call_{uuid4().hex[:8]}",
                            "type": "function",
                            "function": {
                                "name": tc.get("name", "") or "",
                                "arguments": json.dumps({"_raw": tc}, ensure_ascii=False),
                            },
                        }
                    )

            result: Dict[str, Any] = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": text,
                            **({"tool_calls": openai_tool_calls} if openai_tool_calls else {}),
                        },
                        "finish_reason": "tool_calls" if openai_tool_calls else "stop",
                    }
                ],
                "model": model_name,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }

            if self.cache and user_input and not has_tools and not has_tool_results:
                self.cache.set(
                    prompt=user_input,
                    response=result,
                    model=model_name,
                    context_hash=context_hash,
                )

            return result

        except Exception as e:
            print(f"[L.U.N.A. LLM API] Gemini API 호출 중 오류 발생: {e}")

            if user_prompt:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=[gx.Content(role="user", parts=[gx.Part(text=user_prompt)])],
                    )
                    text, _ = self._extract_gemini_output(response)
                    if not text:
                        fallback_txt = getattr(response, "text", None) or getattr(
                            response, "content", None
                        )
                        text = str(fallback_txt) if fallback_txt is not None else str(response)

                    return {
                        "choices": [
                            {
                                "message": {"role": "assistant", "content": text},
                                "finish_reason": "stop",
                            }
                        ],
                        "model": model_name,
                        "usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                        },
                    }
                except Exception as fallback_error:
                    print(f"[L.U.N.A. LLM API] 폴백 호출도 실패: {fallback_error}")
                    return {"error": str(fallback_error)}

            return {"error": str(e)}
    
    def _generate_gemini_stream(
        self,
        client: genai.Client,
        model_name: str,
        contents: List[gx.Content],
        gen_config: Optional[gx.GenerateContentConfig] = None,
        user_input: Optional[str] = None,
        context_hash: str = "",
    ):
        """
        Gemini 스트리밍 응답 제너레이터.
        스트림이 끝나면 full_text를 캐시에 저장.
        """
        try:
            kwargs: Dict[str, Any] = {"model": model_name, "contents": contents}
            if gen_config:
                kwargs["config"] = gen_config

            response_stream = client.models.generate_content_stream(**kwargs)
            full_text = ""

            for chunk in response_stream:
                chunk_text = getattr(chunk, "text", None)
                if isinstance(chunk_text, str) and chunk_text:
                    full_text += chunk_text

                    yield {
                        "choices": [
                            {
                                "delta": {"role": "assistant", "content": chunk_text},
                                "finish_reason": None,
                            }
                        ],
                        "model": model_name,
                    }

            yield {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "model": model_name,
            }

            if self.cache and user_input and full_text:
                complete_response = {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": full_text},
                            "finish_reason": "stop",
                        }
                    ],
                    "model": model_name,
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                }
                self.cache.set(
                    prompt=user_input,
                    response=complete_response,
                    model=model_name,
                    context_hash=context_hash,
                )

        except Exception as e:
            print(f"[L.U.N.A. LLM API] Gemini 스트리밍 중 오류 발생: {e}")
            yield {"error": str(e)}


    def get_available_providers(self) -> List[str]:
        """사용 가능한 API 제공자 목록을 반환합니다."""
        return list(self.clients.keys())