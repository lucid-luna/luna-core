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
    LLMContext
)
from .token_utils import estimate_tokens_for_messages, estimate_tokens_for_text


class MemoryService:
    """
    대화 메모리 관리 서비스 (SQLite 기반)
    
    기존 파일 기반 API와 호환되도록 설계됨
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
        # Token budget for assembled LLM context. If set to None or 0, uses turns-based limit only.
        self.max_context_tokens = max_context_tokens
        self.summary_threshold = summary_threshold
        self.enable_auto_summary = enable_auto_summary
        self.llm_service = llm_service
        self.user_id = user_id
        self.session_id = session_id
        
        # DB 초기화
        db_path = f"{memory_dir}/luna.db"
        self.db_manager = get_db_manager(db_path)
        self.repository = MemoryRepository(self.db_manager)
        
        print(f"[MemoryService] SQLite 기반 초기화 완료 (요약 임계값: {summary_threshold}턴, 자동 요약: {enable_auto_summary})")
    
    # ==================== 기존 API 호환 메서드 ====================
    
    def add_entry(self, user_input: str, assistant_response: str, metadata: Optional[Dict[str, Any]] = None):
        """
        새로운 대화를 메모리에 추가 (기존 API 호환)
        
        Args:
            user_input: 사용자 입력
            assistant_response: 어시스턴트 응답
            metadata: 추가 메타데이터 (감정, 인텐트 등)
        """
        # 메타데이터에서 정보 추출
        emotion = metadata.get("emotion") if metadata else None
        intent = metadata.get("intent") if metadata else None
        processing_time = metadata.get("processing_time") if metadata else None
        cached = metadata.get("cached", False) if metadata else False
        
        # DB에 저장
        conv = ConversationCreate(
            user_id=self.user_id,
            session_id=self.session_id,
            user_message=user_input,
            assistant_message=assistant_response,
            emotion=emotion,
            intent=intent,
            processing_time=processing_time,
            cached=cached,
            metadata=metadata
        )
        
        self.repository.create_conversation(conv)
        
        # 자동 요약 체크
        if self.enable_auto_summary and self.llm_service:
            self._check_and_summarize()
    
    def get_context_for_llm(self, include_system_prompt: bool = False) -> List[Dict[str, str]]:
        """
        LLM에 전달할 대화 컨텍스트 생성 (기존 API 호환)
        
        Args:
            include_system_prompt: 시스템 프롬프트 포함 여부 (사용 안 함)
            
        Returns:
            List[Dict[str, str]]: OpenAI 형식 메시지 리스트
        """
        messages: List[Dict[str, str]] = []

        # 최신 요약 가져오기
        summary = self.repository.get_latest_summary(self.user_id, self.session_id)
        if summary:
            messages.append({
                "role": "system",
                "content": f"[이전 대화 요약]\n{summary.content}"
            })

        # Fetch a reasonably large window (we'll trim by token budget)
        fetch_limit = max(self.max_context_turns * 4, 100)
        conversations = self.repository.get_conversations(
            user_id=self.user_id,
            session_id=self.session_id,
            limit=fetch_limit
        )

        # 시간순 정렬 (오래된 것부터)
        conversations.reverse()

        # If no token budget configured, fall back to turns-based behavior
        if not self.max_context_tokens or self.max_context_tokens <= 0:
            recent = conversations[-self.max_context_turns:]
            recent.reverse()
            for conv in recent:
                messages.append({"role": "user", "content": conv.user_message})
                messages.append({"role": "assistant", "content": conv.assistant_message})
            return messages

        # Build messages under token budget. Start with any existing messages (e.g., summary)
        current_tokens = estimate_tokens_for_messages(messages)

        # We want to keep the chronological order, so iterate conversations oldest->newest
        for conv in conversations:
            # Build the two messages we would add
            user_msg = {"role": "user", "content": conv.user_message}
            assistant_msg = {"role": "assistant", "content": conv.assistant_message}

            # Estimate tokens if we add both messages
            add_tokens = estimate_tokens_for_messages([user_msg, assistant_msg])

            # If adding these would exceed the budget, skip them. However, ensure we include at least the most recent turn.
            if current_tokens + add_tokens > self.max_context_tokens:
                # Check if messages is empty (only possible if summary was too large) — then skip
                # If we're close to the end (this is one of the most recent turns), consider forcing inclusion of the last turn
                continue

            messages.append(user_msg)
            messages.append(assistant_msg)
            current_tokens += add_tokens

        # If we ended up with no conversation messages (e.g., budget too small), ensure at least the most recent turn is present
        if len(messages) <= (1 if summary else 0):
            latest = self.repository.get_conversations(
                user_id=self.user_id,
                session_id=self.session_id,
                limit=1
            )
            if latest:
                conv = latest[0]
                messages.append({"role": "user", "content": conv.user_message})
                messages.append({"role": "assistant", "content": conv.assistant_message})

        # Finally, if we have more turns than max_context_turns, trim older turns while preserving summary
        # Count only user+assistant pairs
        pairs = (len(messages) - (1 if summary else 0)) // 2
        if pairs > self.max_context_turns:
            # Keep the most recent max_context_turns pairs
            preserve_prefix = messages[: (1 if summary else 0)]
            recent_pairs = messages[-(self.max_context_turns * 2):]
            messages = preserve_prefix + recent_pairs

        return messages
    
    def get_recent_summary(self, count: int = 5) -> str:
        """
        최근 대화 요약 텍스트 반환 (기존 API 호환)
        
        Args:
            count: 표시할 대화 수
            
        Returns:
            str: 대화 요약 텍스트
        """
        conversations = self.repository.get_conversations(
            user_id=self.user_id,
            session_id=self.session_id,
            limit=count
        )
        
        if not conversations:
            return "(대화 내역 없음)"
        
        # 시간순 정렬
        conversations.reverse()
        
        summary_parts = []
        for i, conv in enumerate(conversations, 1):
            user_text = conv.user_message[:50] + "..." if len(conv.user_message) > 50 else conv.user_message
            assistant_text = conv.assistant_message[:50] + "..." if len(conv.assistant_message) > 50 else conv.assistant_message
            summary_parts.append(f"{i}. User: {user_text}\n   Assistant: {assistant_text}")
        
        return "\n\n".join(summary_parts)
    
    def clear_memory(self):
        """모든 메모리 삭제 (기존 API 호환)"""
        deleted = self.repository.delete_conversations(
            user_id=self.user_id,
            session_id=self.session_id
        )
        print(f"[MemoryService] 메모리 삭제 완료 ({deleted}개)")
    
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
            "first_conversation": stats.first_conversation.isoformat() if stats.first_conversation else None,
            "last_conversation": stats.last_conversation.isoformat() if stats.last_conversation else None,
            "context_window": self.max_context_turns,
            "max_stored": self.max_entries,
            "auto_summary_enabled": self.enable_auto_summary,
            "summary_threshold": self.summary_threshold
        }
    
    def force_summarize(self) -> bool:
        """
        수동 요약 실행 (기존 API 호환)
        
        Returns:
            bool: 요약 성공 여부
        """
        print("[MemoryService] 수동 요약 실행")
        return self._summarize_conversations()
    
    def get_summary(self) -> Optional[str]:
        """
        현재 저장된 요약 반환 (기존 API 호환)
        
        Returns:
            str: 요약 텍스트 (없으면 None)
        """
        summary = self.repository.get_latest_summary(self.user_id, self.session_id)
        return summary.content if summary else None
    
    # ==================== 내부 메서드 (요약 로직) ====================
    
    def _check_and_summarize(self):
        """대화 수 확인 후 자동 요약 실행"""
        stats = self.repository.get_stats(self.user_id, self.session_id)
        
        if stats.total_conversations >= self.summary_threshold:
            print(f"[MemoryService] 대화 {stats.total_conversations}개 도달 → 자동 요약 실행")
            self._summarize_conversations()
    
    def _summarize_conversations(self) -> bool:
        """
        오래된 대화를 요약하고 DB 정리
        
        Returns:
            bool: 요약 성공 여부
        """
        # 모든 대화 가져오기
        all_conversations = self.repository.get_conversations(
            user_id=self.user_id,
            session_id=self.session_id,
            limit=1000  # 충분히 큰 값
        )
        
        # 시간순 정렬 (오래된 것부터)
        all_conversations.reverse()
        
        if len(all_conversations) <= self.max_context_turns:
            print("[MemoryService] 대화가 충분하지 않아 요약을 건너뜁니다.")
            return False
        
        # 최근 대화는 유지, 오래된 대화만 요약
        recent_conversations = all_conversations[-self.max_context_turns:]
        old_conversations = all_conversations[:-self.max_context_turns]
        
        if not old_conversations:
            return False
        
        # 기존 요약 가져오기
        existing_summary = self.repository.get_latest_summary(self.user_id, self.session_id)
        
        # 요약 생성
        new_summary_text = self._generate_summary(old_conversations, existing_summary)
        
        if new_summary_text:
            # 요약 저장
            summary = SummaryCreate(
                user_id=self.user_id,
                session_id=self.session_id,
                content=new_summary_text,
                summarized_turns=len(old_conversations),
                start_conversation_id=old_conversations[0].id,
                end_conversation_id=old_conversations[-1].id
            )
            self.repository.create_summary(summary)
            
            # 요약된 대화 삭제 (선택적)
            # for conv in old_conversations:
            #     self.repository.delete_conversation(conv.id)
            
            print(f"[MemoryService] 요약 완료 ({len(old_conversations)}턴 → 요약)")
            return True
        
        return False
    
    def _generate_summary(
        self,
        conversations: List[ConversationResponse],
        existing_summary: Optional[SummaryResponse] = None
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
            print("[MemoryService] LLM 서비스가 없어 요약을 생성할 수 없습니다.")
            return None
        
        # 대화 내용 포맷팅
        conversation_text = []
        for conv in conversations:
            conversation_text.append(f"User: {conv.user_message}")
            conversation_text.append(f"Assistant: {conv.assistant_message}")
        
        conversation_str = "\n".join(conversation_text)
        
        # 요약 프롬프트
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
            # LLM에 요약 요청
            target = list(self.llm_service.get_available_targets())[0]
            response = self.llm_service.generate(
                target=target,
                system_prompt="당신은 대화 내용을 간결하고 정확하게 요약하는 전문가입니다.",
                user_prompt=prompt
            )
            
            # 응답 파싱
            if isinstance(response, dict) and "choices" in response:
                summary_text = response["choices"][0]["message"]["content"]
                return summary_text.strip()
            
            return None
        
        except Exception as e:
            print(f"[MemoryService] 요약 생성 중 오류: {e}")
            return None
    
    # ==================== 새로운 기능 (DB 기반) ====================
    
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
            user_id=self.user_id,
            session_id=self.session_id,
            limit=limit,
            offset=offset
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
            limit=limit
        )
        
        results, _ = self.repository.search_conversations(search)
        return results
