# ====================================================================
#  File: services/memory/memory_service.py
# ====================================================================
"""
메모리 서비스 (SQLite 기반, 기존 API 호환)
"""

from typing import List, Dict, Any, Optional
from datetime import datetime

from .database import get_db_manager, DatabaseManager
from .repository import MemoryRepository
from .models import (
    ConversationCreate,
    ConversationResponse,
    SummaryCreate,
    SummaryResponse,
    LLMContext,
    CoreMemoryCreate,
    CoreMemoryResponse,
    WorkingMemoryCreate,
    WorkingMemoryResponse,
    CoreMemoryCategory,
)
from .token_utils import estimate_tokens_for_messages, estimate_tokens_for_text


class MemoryService:
    """
    대화 메모리 관리 서비스
    """
    
    def __init__(
        self,
        memory_dir: str = "./memory",
        max_entries: int = 50,
        max_context_turns: int = 6,
        max_context_tokens: int = 1500,
        summary_threshold: int = 20,
        enable_auto_summary: bool = True,
        llm_service=None,
        user_id: str = "default",
        session_id: str = "default",
    ):
        """
        메모리 서비스 초기화

        Args:
            memory_dir: 메모리 디렉토리 (SQLite 파일 위치)
            max_entries: 저장할 최대 대화 수 (사용 안 함, DB가 관리)
            max_context_turns: LLM에 전달할 최대 대화 턴 수
            summary_threshold: 자동 요약 임계값 (대화 수)
            enable_auto_summary: 자동 요약 활성화 여부
            llm_service: LLM 서비스 인스턴스
            user_id: 사용자 ID (기본: "default")
            session_id: 세션 ID (기본: "default")
        """
        self.memory_dir = memory_dir
        self.max_entries = max_entries
        self.max_context_turns = max_context_turns
        self.max_context_tokens = max_context_tokens
        self.summary_threshold = summary_threshold
        self.enable_auto_summary = enable_auto_summary
        self.llm_service = llm_service
        self.user_id = user_id
        self.session_id = session_id

        db_path = f"{memory_dir}/luna.db"
        self.db_manager: DatabaseManager = get_db_manager(db_path)
        self.repository = MemoryRepository(self.db_manager)

        print(
            f"[L.U.N.A. MemoryService] SQLite 기반 초기화 완료 "
            f"(요약 임계값: {summary_threshold}턴, 자동 요약: {enable_auto_summary})"
        )
        
    # ==================== 기존 API 호환 메서드 ====================

    def add_entry(
        self,
        user_input: str,
        assistant_response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        새로운 대화를 메모리에 추가 (기존 API 호환)

        Args:
            user_input: 사용자 입력
            assistant_response: 어시스턴트 응답
            metadata: 추가 메타데이터 (감정, 인텐트 등)
        """
        emotion = metadata.get("emotion") if metadata else None
        intent = metadata.get("intent") if metadata else None
        processing_time = metadata.get("processing_time") if metadata else None
        cached = metadata.get("cached", False) if metadata else False

        conv = ConversationCreate(
            user_id=self.user_id,
            session_id=self.session_id,
            user_message=user_input,
            assistant_message=assistant_response,
            emotion=emotion,
            intent=intent,
            processing_time=processing_time,
            cached=cached,
            metadata=metadata,
        )

        self.repository.create_conversation(conv)

        # 만료된 단기 메모리 정리 (10번째 대화마다)
        stats = self.repository.get_stats(self.user_id, self.session_id)
        if stats.total_conversations % 10 == 0:
            self.cleanup_expired_memories()

        # 중요 정보 패턴 감지 시에만 메모리 추출 (백그라운드로 실행하여 응답 지연 방지)
        if self.llm_service and self._should_extract_memory(user_input):
            import threading

            def extract_in_background():
                try:
                    self.extract_and_store_memories(user_input, assistant_response)
                except Exception as e:
                    print(f"[MemoryService] 메모리 추출 실패 (무시): {e}")

            thread = threading.Thread(target=extract_in_background, daemon=True)
            thread.start()

        if self.enable_auto_summary and self.llm_service:
            self._check_and_summarize()

    def _should_extract_memory(self, user_input: str) -> bool:
        """
        메모리 추출이 필요한지 키워드 패턴으로 판단

        - core: 사용자 정보 / 선호 / 관계 / 거주지 / 프로젝트 전반
        - working: 지금/오늘/요즘 하는 작업, 임시 목적

        너무 자주 트리거되지 않도록 "길이 5 이하 단문"은 필터링.
        """
        text = (user_input or "").strip()
        if not text:
            return False

        # 아주 짧은 단문은 스킵 (인사, 한 단어 감상 등)
        if len(text) <= 5:
            return False

        # 장기 메모리 트리거 패턴 (사용자 정보)
        core_patterns = [
            # 자기소개 / 정체성
            "내 이름", "제 이름", "나는 ", "저는 ", "내가 누구", "나는 누구",
            # 선호 / 취향
            "좋아하", "싫어하", "제일 좋아", "최애", "my favorite",
            # 직업 / 학교 / 거주
            "직업", "일하", "회사", "학교", "대학", "살고 있", "사는 곳", "거주",
            # 관계 / 호칭
            "불러줘", "불러 줘", "호칭", "뭐라고 불러", "어떻게 불러",
            # 기억 요청
            "기억해", "기억해줘", "잊지마", "잊지 마", "알아둬", "알아 둬",
        ]

        # 단기 메모리 트리거 패턴 (현재 작업)
        working_patterns = [
            "작업", "프로젝트", "만들", "개발",
            "지금 ", "오늘 ", "요즘 ",
            "하는 중", "하고 있", "진행 중", "정리해줘", "정리 해줘",
        ]

        lower = text.lower()

        for pattern in core_patterns:
            if pattern in text or pattern in lower:
                return True

        for pattern in working_patterns:
            if pattern in text or pattern in lower:
                return True

        return False

    def get_context_for_llm(self, include_system_prompt: bool = False) -> List[Dict[str, str]]:
        """
        LLM에 전달할 대화 컨텍스트 생성 (기존 API 호환)

        Args:
            include_system_prompt: 시스템 프롬프트 포함 여부 (사용 안 함)

        Returns:
            List[Dict[str, str]]: OpenAI 형식 메시지 리스트
        """
        messages: List[Dict[str, str]] = []

        # 1) 최신 요약을 맨 앞 system으로 넣는다
        summary = self.repository.get_latest_summary(self.user_id, self.session_id)
        if summary:
            messages.append(
                {
                    "role": "system",
                    "content": f"[이전 대화 요약]\n{summary.content}",
                }
            )

        # 2) 최근 대화들을 토큰 한도 안에서 뒤에서부터 쌓기
        fetch_limit = max(self.max_context_turns * 4, 100)
        conversations = self.repository.get_conversations(
            user_id=self.user_id,
            session_id=self.session_id,
            limit=fetch_limit,
        )

        # 최근 순서대로 정렬 (timestamp DESC로 가져왔으니 뒤집기)
        conversations.reverse()

        # max_context_tokens 비활성화 시, 단순 recent n턴
        if not self.max_context_tokens or self.max_context_tokens <= 0:
            recent = conversations[-self.max_context_turns :]
            recent.reverse()
            for conv in recent:
                messages.append({"role": "user", "content": conv.user_message})
                messages.append({"role": "assistant", "content": conv.assistant_message})
            return messages

        # 토큰 단위로 잘라 넣기
        current_tokens = estimate_tokens_for_messages(messages)

        for conv in conversations:
            user_msg = {"role": "user", "content": conv.user_message}
            assistant_msg = {"role": "assistant", "content": conv.assistant_message}

            add_tokens = estimate_tokens_for_messages([user_msg, assistant_msg])

            if current_tokens + add_tokens > self.max_context_tokens:
                continue

            messages.append(user_msg)
            messages.append(assistant_msg)
            current_tokens += add_tokens

        # 혹시 아무것도 안 들어갔으면 가장 최신 한 턴이라도 넣기
        if len(messages) <= (1 if summary else 0):
            latest = self.repository.get_conversations(
                user_id=self.user_id,
                session_id=self.session_id,
                limit=1,
            )
            if latest:
                conv = latest[0]
                messages.append({"role": "user", "content": conv.user_message})
                messages.append({"role": "assistant", "content": conv.assistant_message})

        # 턴 수 제한(max_context_turns) 초과 시, 뒤쪽 n턴만 유지
        pairs = (len(messages) - (1 if summary else 0)) // 2
        if pairs > self.max_context_turns:
            preserve_prefix = messages[: (1 if summary else 0)]
            recent_pairs = messages[-(self.max_context_turns * 2) :]
            messages = preserve_prefix + recent_pairs

        return messages

    def get_recent_summary(self, count: int = 5) -> str:
        """
        최근 대화 요약 텍스트 반환

        Args:
            count: 표시할 대화 수

        Returns:
            str: 대화 요약 텍스트
        """
        conversations = self.repository.get_conversations(
            user_id=self.user_id,
            session_id=self.session_id,
            limit=count,
        )

        if not conversations:
            return "(대화 내역 없음)"

        conversations.reverse()

        summary_parts = []
        for i, conv in enumerate(conversations, 1):
            user_text = (
                conv.user_message[:50] + "..."
                if len(conv.user_message) > 50
                else conv.user_message
            )
            assistant_text = (
                conv.assistant_message[:50] + "..."
                if len(conv.assistant_message) > 50
                else conv.assistant_message
            )
            summary_parts.append(
                f"{i}. User: {user_text}\n   Assistant: {assistant_text}"
            )

        return "\n\n".join(summary_parts)

    def clear_memory(self):
        """모든 메모리 삭제 (기존 API 호환)"""
        deleted = self.repository.delete_conversations(
            user_id=self.user_id, session_id=self.session_id
        )
        print(f"[L.U.N.A. MemoryService] 메모리 삭제 완료 ({deleted}개)")

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        메모리 통계 반환 (기존 API 호환)

        Returns:
            Dict[str, Any]: 통계 정보
        """
        stats = self.repository.get_stats(self.user_id, self.session_id)

        return {
            "total_entries": stats.total_conversations + stats.total_summaries,
            "conversations": stats.total_conversations,
            "summaries": stats.total_summaries,
            "first_conversation": stats.first_conversation.isoformat()
            if stats.first_conversation
            else None,
            "last_conversation": stats.last_conversation.isoformat()
            if stats.last_conversation
            else None,
            "context_window": self.max_context_turns,
            "max_stored": self.max_entries,
            "auto_summary_enabled": self.enable_auto_summary,
            "summary_threshold": self.summary_threshold,
        }

    def force_summarize(self) -> bool:
        """
        수동 요약 실행 (기존 API 호환)

        Returns:
            bool: 요약 성공 여부
        """
        print("[L.U.N.A. MemoryService] 수동 요약 실행")
        return self._summarize_conversations()

    def get_summary(self) -> Optional[str]:
        """
        현재 저장된 요약 반환 (기존 API 호환)

        Returns:
            str: 요약 텍스트 (없으면 None)
        """
        summary = self.repository.get_latest_summary(self.user_id, self.session_id)
        return summary.content if summary else None

    def _check_and_summarize(self):
        """대화 수 확인 후 자동 요약 실행"""
        stats = self.repository.get_stats(self.user_id, self.session_id)

        if stats.total_conversations >= self.summary_threshold:
            print(
                f"[L.U.N.A. MemoryService] 대화 {stats.total_conversations}개 도달 → 자동 요약 실행"
            )
            self._summarize_conversations()

    def _summarize_conversations(self) -> bool:
        """
        오래된 대화를 요약하고 DB 정리

        Returns:
            bool: 요약 성공 여부
        """
        all_conversations = self.repository.get_conversations(
            user_id=self.user_id,
            session_id=self.session_id,
            limit=1000,
        )

        all_conversations.reverse()

        if len(all_conversations) <= self.max_context_turns:
            print("[L.U.N.A. MemoryService] 대화가 충분하지 않아 요약을 건너뜁니다.")
            return False

        # 최근 n턴은 그대로 두고, 이전 것만 요약
        recent_conversations = all_conversations[-self.max_context_turns :]
        old_conversations = all_conversations[: -self.max_context_turns]

        if not old_conversations:
            return False

        existing_summary = self.repository.get_latest_summary(
            self.user_id, self.session_id
        )

        new_summary_text = self._generate_summary(old_conversations, existing_summary)

        if new_summary_text:
            summary = SummaryCreate(
                user_id=self.user_id,
                session_id=self.session_id,
                content=new_summary_text,
                summarized_turns=len(old_conversations),
                start_conversation_id=old_conversations[0].id,
                end_conversation_id=old_conversations[-1].id,
            )
            self.repository.create_summary(summary)

            print(
                f"[L.U.N.A. MemoryService] 요약 완료 ({len(old_conversations)}턴 → 요약)"
            )
            return True

        return False

    def _generate_summary(
        self,
        conversations: List[ConversationResponse],
        existing_summary: Optional[SummaryResponse] = None,
    ) -> Optional[str]:
        """
        LLM을 사용하여 대화 요약 생성

        Args:
            conversations: 요약할 대화 리스트
            existing_summary: 기존 요약 (있으면 업데이트)

        Returns:
            str: 생성된 요약 텍스트
        """
        if not self.llm_service:
            print("[L.U.N.A. MemoryService] LLM 서비스가 없어 요약을 생성할 수 없습니다.")
            return None

        conversation_text = []
        for conv in conversations:
            conversation_text.append(f"User: {conv.user_message}")
            conversation_text.append(f"Assistant: {conv.assistant_message}")

        conversation_str = "\n".join(conversation_text)

        if existing_summary:
            prompt = f"""기존 요약:
{existing_summary.content}

추가 대화 내역:
{conversation_str}

위의 기존 요약에 추가 대화 내용을 반영하여 업데이트된 요약을 작성해주세요.
요약에는 다음 내용을 포함하세요:
- 사용자의 주요 관심사와 목적
- 중요한 결정 사항이나 합의 내용
- 반복되는 주제나 패턴
- 향후 대화에 참조가 필요한 정보

간결하고 핵심적인 내용만 포함하여 3-5문장으로 작성해주세요."""
        else:
            prompt = f"""다음 대화 내역을 요약해주세요:

{conversation_str}

요약에는 다음 내용을 포함하세요:
- 사용자의 주요 관심사와 목적
- 중요한 결정 사항이나 합의 내용
- 반복되는 주제나 패턴
- 향후 대화에 참조가 필요한 정보

간결하고 핵심적인 내용만 포함하여 3-5문장으로 작성해주세요."""

        try:
            target_list = list(self.llm_service.get_available_targets())
            if not target_list:
                print("[MemoryService] 사용 가능한 LLM target이 없습니다.")
                return None
            target = target_list[0]

            response = self.llm_service.generate(
                target=target,
                system_prompt="당신은 대화 내용을 간결하고 정확하게 요약하는 전문가입니다.",
                user_prompt=prompt,
            )

            if isinstance(response, dict) and "choices" in response:
                summary_text = response["choices"][0]["message"]["content"]
                return (summary_text or "").strip() or None

            return None

        except Exception as e:
            print(f"[MemoryService] 요약 생성 중 오류: {e}")
            return None

    # ====================================================================
    #  대화 조회 / 검색 (새 기능)
    # ====================================================================

    def get_conversations(self, limit: int = 50, offset: int = 0) -> List[ConversationResponse]:
        """
        대화 목록 조회 (새 기능)

        Args:
            limit: 최대 개수
            offset: 시작 위치

        Returns:
            List[ConversationResponse]: 대화 목록
        """
        return self.repository.get_conversations(
            user_id=self.user_id, session_id=self.session_id, limit=limit, offset=offset
        )

    def search_conversations(self, keyword: str, limit: int = 50) -> List[ConversationResponse]:
        """
        키워드로 대화 검색 (새 기능)

        Args:
            keyword: 검색 키워드
            limit: 최대 개수

        Returns:
            List[ConversationResponse]: 검색 결과
        """
        from .models import ConversationSearchRequest

        search = ConversationSearchRequest(
            user_id=self.user_id,
            session_id=self.session_id,
            keyword=keyword,
            limit=limit,
        )

        results, _ = self.repository.search_conversations(search)
        return results

    # ====================================================================
    #  장기 메모리 (Core Memory) API
    # ====================================================================

    def add_core_memory(
        self,
        category: str,
        key: str,
        value: str,
        importance: int = 5,
        source: Optional[str] = None,
    ) -> CoreMemoryResponse:
        """
        장기 메모리 추가/업데이트

        Args:
            category: 카테고리 (user_info, preferences, projects, relationships, facts)
            key: 고유 키 (예: "name", "favorite_color")
            value: 저장할 값
            importance: 중요도 (1-10)
            source: 출처 정보

        Returns:
            CoreMemoryResponse: 저장된 메모리
        """
        memory = CoreMemoryCreate(
            user_id=self.user_id,
            category=category,
            key=key,
            value=value,
            importance=importance,
            source=source,
        )
        return self.repository.create_or_update_core_memory(memory)

    def get_core_memories(
        self,
        category: Optional[str] = None,
        min_importance: int = 1,
    ) -> List[CoreMemoryResponse]:
        """
        장기 메모리 조회

        Args:
            category: 카테고리 필터 (선택)
            min_importance: 최소 중요도

        Returns:
            List[CoreMemoryResponse]: 메모리 목록
        """
        return self.repository.get_core_memories(
            user_id=self.user_id, category=category, min_importance=min_importance
        )

    def get_core_memory(self, category: str, key: str) -> Optional[CoreMemoryResponse]:
        """특정 장기 메모리 조회"""
        return self.repository.get_core_memory_by_key(self.user_id, category, key)

    def update_core_memory(
        self,
        memory_id: int,
        value: Optional[str] = None,
        importance: Optional[int] = None,
    ) -> Optional[CoreMemoryResponse]:
        """장기 메모리 수정"""
        return self.repository.update_core_memory(memory_id, value=value, importance=importance)

    def delete_core_memory(self, category: str, key: str) -> bool:
        """장기 메모리 삭제 (category/key로)"""
        return self.repository.delete_core_memory(self.user_id, category, key)

    def delete_core_memory_by_id(self, memory_id: int) -> bool:
        """장기 메모리 삭제 (ID로)"""
        return self.repository.delete_core_memory_by_id(memory_id)

    def get_core_memories_for_context(self) -> str:
        """
        LLM 컨텍스트용 장기 메모리 텍스트 생성

        Returns:
            str: 포맷된 장기 메모리 텍스트
        """
        memories = self.get_core_memories(min_importance=3)

        if not memories:
            return ""

        sections: Dict[str, List[str]] = {}
        for mem in memories:
            sections.setdefault(mem.category, []).append(f"- {mem.key}: {mem.value}")

        result = ["[사용자 정보 - 장기 기억]"]

        category_labels = {
            CoreMemoryCategory.USER_INFO: "기본 정보",
            CoreMemoryCategory.PREFERENCES: "선호도",
            CoreMemoryCategory.PROJECTS: "프로젝트",
            CoreMemoryCategory.RELATIONSHIPS: "관계",
            CoreMemoryCategory.FACTS: "중요 사실",
        }

        for cat, items in sections.items():
            label = category_labels.get(cat, cat)
            result.append(f"\n{label}:")
            result.extend(items)

        return "\n".join(result)

    # ====================================================================
    #  단기 메모리 (Working Memory) API
    # ====================================================================

    def add_working_memory(
        self,
        topic: str,
        content: str,
        importance: int = 3,
        expires_days: int = 3,
    ) -> WorkingMemoryResponse:
        """
        단기 메모리 추가

        Args:
            topic: 주제 (예: "노션 페이지 작업")
            content: 내용
            importance: 중요도 (1-10)
            expires_days: 만료까지 일수 (기본: 3일)

        Returns:
            WorkingMemoryResponse: 저장된 메모리
        """
        from datetime import timedelta

        expires_at = datetime.now() + timedelta(days=expires_days)

        memory = WorkingMemoryCreate(
            user_id=self.user_id,
            session_id=self.session_id,
            topic=topic,
            content=content,
            importance=importance,
            expires_at=expires_at,
        )
        return self.repository.create_working_memory(memory)

    def get_working_memories(
        self,
        topic: Optional[str] = None,
        include_expired: bool = False,
    ) -> List[WorkingMemoryResponse]:
        """
        단기 메모리 조회

        Args:
            topic: 주제 필터 (선택)
            include_expired: 만료된 것도 포함

        Returns:
            List[WorkingMemoryResponse]: 메모리 목록
        """
        return self.repository.get_working_memories(
            user_id=self.user_id,
            session_id=self.session_id,
            include_expired=include_expired,
            topic=topic,
        )

    def cleanup_expired_memories(self) -> int:
        """만료된 단기 메모리 정리"""
        deleted = self.repository.delete_expired_working_memories()
        if deleted > 0:
            print(f"[MemoryService] 만료된 단기 메모리 {deleted}개 삭제")
        return deleted

    def extend_working_memory(self, memory_id: int, days: int = 3) -> bool:
        """단기 메모리 만료 연장"""
        return self.repository.extend_working_memory(memory_id, days)

    def delete_working_memory(self, memory_id: int) -> bool:
        """단기 메모리 삭제"""
        return self.repository.delete_working_memory(memory_id)

    def get_working_memories_for_context(self) -> str:
        """
        LLM 컨텍스트용 단기 메모리 텍스트 생성

        Returns:
            str: 포맷된 단기 메모리 텍스트
        """
        memories = self.get_working_memories(include_expired=False)

        if not memories:
            return ""

        result = ["[최근 주제 - 단기 기억]"]

        now = datetime.now()
        for mem in memories:
            days_left = (mem.expires_at - now).days
            if days_left < 0:
                days_left = 0
            result.append(f"- {mem.topic}: {mem.content} (유효: {days_left}일)")

        return "\n".join(result)

    # ====================================================================
    #  통합 컨텍스트 생성
    # ====================================================================

    def get_full_context_for_llm(self) -> List[Dict[str, str]]:
        """
        장기/단기 메모리를 포함한 전체 LLM 컨텍스트 생성

        Returns:
            List[Dict[str, str]]: OpenAI 형식 메시지 리스트
        """
        messages: List[Dict[str, str]] = []

        # 1. 장기 메모리 (사용자 정보)
        core_context = self.get_core_memories_for_context()
        if core_context:
            messages.append({"role": "system", "content": core_context})

        # 2. 단기 메모리 (최근 주제)
        working_context = self.get_working_memories_for_context()
        if working_context:
            messages.append({"role": "system", "content": working_context})

        # 3. 기존 대화 컨텍스트 (요약 + 최근 턴들)
        conversation_context = self.get_context_for_llm()
        messages.extend(conversation_context)

        return messages

    # ====================================================================
    #  LLM 기반 메모리 추출
    # ====================================================================

    def extract_and_store_memories(self, user_input: str, assistant_response: str):
        """
        대화에서 중요 정보를 추출하여 메모리에 저장 (LLM 사용)

        Args:
            user_input: 사용자 입력
            assistant_response: 어시스턴트 응답
        """
        if not self.llm_service:
            return

        prompt = f"""다음 대화에서 저장할 만한 중요한 정보가 있는지 분석해주세요.

사용자: {user_input}
어시스턴트: {assistant_response}

다음 형식으로 JSON 응답해주세요:
{{
  "core_memories": [
    {{
      "category": "user_info|preferences|projects|relationships|facts",
      "key": "키",
      "value": "값",
      "importance": 1-10
    }}
  ],
  "working_memories": [
    {{
      "topic": "주제",
      "content": "내용",
      "importance": 1-10
    }}
  ]
}}

조건:
- 저장할 정보가 없으면 각 배열을 빈 배열로 두세요.
- 이름, 직업, 관심사 등 장기적으로 중요한 개인정보는 core_memories.user_info 로.
- 선호/취향은 core_memories.preferences 로.
- 현재 작업 중인 것, 일시적 주제는 working_memories 로.
- 사소한 일상 대화는 넣지 마세요.
"""

        try:
            target_list = list(self.llm_service.get_available_targets())
            if not target_list:
                print("[MemoryService] 메모리 추출용 LLM target이 없습니다.")
                return
            target = target_list[0]

            response = self.llm_service.generate(
                target=target,
                system_prompt="당신은 대화에서 중요 정보를 추출하는 분석가입니다. JSON만 응답하세요.",
                user_prompt=prompt,
            )

            if not (isinstance(response, dict) and "choices" in response):
                print("[MemoryService] 메모리 추출 응답 형식이 올바르지 않습니다.")
                return

            import json

            content = response["choices"][0]["message"]["content"] or ""
            raw_content = content

            # ```json ... ``` 또는 ``` ... ``` 래핑 제거
            if "```json" in content:
                try:
                    content = content.split("```json", 1)[1].split("```", 1)[0]
                except Exception:
                    pass
            elif "```" in content:
                try:
                    content = content.split("```", 1)[1].split("```", 1)[0]
                except Exception:
                    pass

            content = content.strip()
            if not content:
                print("[MemoryService] 메모리 추출 응답이 비어 있습니다.")
                return

            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                print(
                    "[MemoryService] 메모리 추출 JSON 파싱 실패, 원본 응답 일부:\n"
                    f"{raw_content[:300]}"
                )
                return

            core_list = data.get("core_memories") or []
            working_list = data.get("working_memories") or []

            # 장기 메모리 저장
            for mem in core_list:
                try:
                    category = mem.get("category") or CoreMemoryCategory.FACTS
                    key = mem.get("key")
                    value = mem.get("value")
                    if not key or not value:
                        continue

                    importance = int(mem.get("importance", 5))
                    importance = max(1, min(10, importance))

                    self.add_core_memory(
                        category=category,
                        key=key,
                        value=value,
                        importance=importance,
                        source=f"conversation_{datetime.now().isoformat()}",
                    )
                    print(f"[MemoryService] 장기 메모리 저장: {category}/{key}")
                except Exception as e:
                    print(f"[MemoryService] 장기 메모리 저장 중 오류: {e}")

            # 단기 메모리 저장
            for mem in working_list:
                try:
                    topic = mem.get("topic")
                    content = mem.get("content")
                    if not topic or not content:
                        continue

                    importance = int(mem.get("importance", 3))
                    importance = max(1, min(10, importance))

                    self.add_working_memory(
                        topic=topic,
                        content=content,
                        importance=importance,
                    )
                    print(f"[MemoryService] 단기 메모리 저장: {topic}")
                except Exception as e:
                    print(f"[MemoryService] 단기 메모리 저장 중 오류: {e}")

        except Exception as e:
            print(f"[MemoryService] 메모리 추출 중 오류: {e}")