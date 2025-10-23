# ====================================================================
#  File: tests/test_memory_service.py
# ====================================================================
"""
MemoryService 토큰 제한 컨텍스트 창 테스트
"""

import pytest
import os
import tempfile
import shutil
from datetime import datetime
from services.memory.memory_service import MemoryService
from services.memory.token_utils import estimate_tokens_for_messages


@pytest.fixture
def temp_memory_dir():
    """임시 메모리 디렉토리 생성"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # 테스트 후 정리
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def memory_service(temp_memory_dir):
    """테스트용 MemoryService 인스턴스"""
    import uuid
    # 각 테스트마다 고유한 session_id 사용
    session_id = f"test_session_{uuid.uuid4().hex[:8]}"
    service = MemoryService(
        memory_dir=temp_memory_dir,
        max_context_turns=6,
        max_context_tokens=500,  # 테스트용 작은 값
        enable_auto_summary=False,  # 자동 요약 비활성화
        user_id="test_user",
        session_id=session_id
    )
    return service


class TestMemoryServiceTokenBudget:
    """토큰 예산 기반 컨텍스트 생성 테스트"""
    
    def test_empty_memory(self, memory_service):
        """빈 메모리에서 컨텍스트 생성"""
        context = memory_service.get_context_for_llm()
        assert context == []
    
    def test_single_conversation(self, memory_service):
        """단일 대화 추가 후 컨텍스트"""
        memory_service.add_entry(
            user_input="안녕하세요",
            assistant_response="네, 안녕하세요!"
        )
        
        context = memory_service.get_context_for_llm()
        assert len(context) == 2  # user + assistant
        assert context[0]["role"] == "user"
        assert context[0]["content"] == "안녕하세요"
        assert context[1]["role"] == "assistant"
        assert context[1]["content"] == "네, 안녕하세요!"
    
    def test_context_under_token_budget(self, memory_service):
        """토큰 예산 내에서 컨텍스트 생성"""
        # 짧은 대화 5개 추가
        for i in range(5):
            memory_service.add_entry(
                user_input=f"질문 {i}",
                assistant_response=f"답변 {i}"
            )
        
        context = memory_service.get_context_for_llm()
        tokens = estimate_tokens_for_messages(context)
        
        # 토큰 예산(500) 이내여야 함
        assert tokens <= memory_service.max_context_tokens
        # 적어도 일부 대화는 포함되어야 함
        assert len(context) >= 2
    
    def test_context_exceeds_budget_truncates(self, memory_service):
        """토큰 예산 초과 시 오래된 대화 제외"""
        # 긴 대화 여러 개 추가
        for i in range(20):
            memory_service.add_entry(
                user_input=f"이것은 질문 {i}번입니다. 좀 더 긴 텍스트를 만들기 위해 여러 단어를 추가합니다.",
                assistant_response=f"이것은 답변 {i}번입니다. 마찬가지로 긴 텍스트로 만들어서 토큰을 많이 사용하도록 합니다."
            )
        
        context = memory_service.get_context_for_llm()
        tokens = estimate_tokens_for_messages(context)
        
        # 토큰 예산 준수
        assert tokens <= memory_service.max_context_tokens
        
        # 최신 대화는 반드시 포함
        assert len(context) >= 2
        
        # 모든 20개 대화를 다 포함하지는 않음
        assert len(context) < 40  # 20 turns * 2 messages
    
    def test_at_least_one_turn_included(self, memory_service):
        """예산이 매우 작아도 최소 1개 턴은 포함"""
        # 매우 작은 토큰 예산
        memory_service.max_context_tokens = 10
        
        # 긴 대화 추가
        memory_service.add_entry(
            user_input="이것은 매우 긴 사용자 입력입니다. " * 20,
            assistant_response="이것은 매우 긴 어시스턴트 응답입니다. " * 20
        )
        
        context = memory_service.get_context_for_llm()
        
        # 예산을 초과하더라도 최소 최근 1개 턴은 포함
        assert len(context) >= 2
    
    def test_turn_limit_still_applies(self, memory_service):
        """토큰 예산 내에서도 max_context_turns 제한 적용"""
        # max_context_turns = 6이므로 최대 6턴까지만
        memory_service.max_context_tokens = 10000  # 매우 큰 예산
        
        # 10개 대화 추가
        for i in range(10):
            memory_service.add_entry(
                user_input=f"질문 {i}",
                assistant_response=f"답변 {i}"
            )
        
        context = memory_service.get_context_for_llm()
        
        # 최대 6턴(12개 메시지)까지만
        assert len(context) <= 12
    
    def test_zero_token_budget_uses_turn_limit(self, memory_service):
        """토큰 예산이 0이면 턴 기반 제한 사용"""
        memory_service.max_context_tokens = 0
        
        # 10개 대화 추가
        for i in range(10):
            memory_service.add_entry(
                user_input=f"질문 {i}",
                assistant_response=f"답변 {i}"
            )
        
        context = memory_service.get_context_for_llm()
        
        # max_context_turns=6이므로 최대 12개 메시지
        assert len(context) == 12
    
    def test_messages_in_chronological_order(self, memory_service):
        """메시지가 시간순으로 정렬됨"""
        for i in range(5):
            memory_service.add_entry(
                user_input=f"질문 {i}",
                assistant_response=f"답변 {i}"
            )
        
        context = memory_service.get_context_for_llm()
        
        # user/assistant 쌍으로 번갈아 나타나야 함
        for i in range(0, len(context), 2):
            if i < len(context):
                assert context[i]["role"] == "user"
            if i + 1 < len(context):
                assert context[i + 1]["role"] == "assistant"


class TestMemoryServiceWithSummary:
    """요약 포함 컨텍스트 테스트"""
    
    def test_summary_included_as_system_message(self, temp_memory_dir):
        """요약이 시스템 메시지로 포함됨"""
        # 요약 생성을 위해 LLM 서비스 목(mock) 필요
        # 간단한 테스트를 위해 수동으로 요약 추가
        service = MemoryService(
            memory_dir=temp_memory_dir,
            max_context_tokens=500,
            user_id="test_user",
            session_id="test_session"
        )
        
        # 대화 추가
        for i in range(3):
            service.add_entry(
                user_input=f"질문 {i}",
                assistant_response=f"답변 {i}"
            )
        
        # 요약 수동 추가
        from services.memory.models import SummaryCreate
        summary = SummaryCreate(
            user_id="test_user",
            session_id="test_session",
            content="이전 대화 요약: 사용자가 여러 질문을 했습니다.",
            summarized_turns=10,
            start_conversation_id=1,
            end_conversation_id=10
        )
        service.repository.create_summary(summary)
        
        context = service.get_context_for_llm()
        
        # 첫 메시지가 시스템 메시지(요약)
        assert len(context) > 0
        assert context[0]["role"] == "system"
        assert "이전 대화 요약" in context[0]["content"]
    
    def test_summary_counts_toward_token_budget(self, temp_memory_dir):
        """요약도 토큰 예산에 포함됨"""
        service = MemoryService(
            memory_dir=temp_memory_dir,
            max_context_tokens=200,  # 작은 예산
            user_id="test_user",
            session_id="test_session"
        )
        
        # 긴 요약 추가
        from services.memory.models import SummaryCreate
        summary = SummaryCreate(
            user_id="test_user",
            session_id="test_session",
            content="이것은 매우 긴 요약 텍스트입니다. " * 30,  # 많은 토큰 사용
            summarized_turns=10,
            start_conversation_id=1,
            end_conversation_id=10
        )
        service.repository.create_summary(summary)
        
        # 대화 추가
        for i in range(10):
            service.add_entry(
                user_input=f"질문 {i}",
                assistant_response=f"답변 {i}"
            )
        
        context = service.get_context_for_llm()
        tokens = estimate_tokens_for_messages(context)
        
        # 요약 포함해도 토큰 예산 준수
        assert tokens <= service.max_context_tokens


class TestMemoryServiceEdgeCases:
    """엣지 케이스 테스트"""
    
    def test_very_long_single_message(self, memory_service):
        """매우 긴 단일 메시지"""
        long_text = "이것은 매우 긴 텍스트입니다. " * 200
        
        memory_service.add_entry(
            user_input=long_text,
            assistant_response=long_text
        )
        
        context = memory_service.get_context_for_llm()
        
        # 예산을 초과해도 최근 대화는 포함
        assert len(context) >= 2
    
    def test_unicode_and_emoji(self, memory_service):
        """유니코드 및 이모지 처리"""
        memory_service.add_entry(
            user_input="안녕하세요! 😊🎉",
            assistant_response="네! 반갑습니다 ✨🌟"
        )
        
        context = memory_service.get_context_for_llm()
        assert len(context) == 2
        assert "😊" in context[0]["content"]
        assert "✨" in context[1]["content"]
    
    def test_metadata_preserved(self, memory_service):
        """메타데이터가 보존됨 (DB에 저장)"""
        metadata = {
            "emotion": "happy",
            "intent": "greeting",
            "cached": False
        }
        
        memory_service.add_entry(
            user_input="안녕하세요",
            assistant_response="네, 안녕하세요!",
            metadata=metadata
        )
        
        # DB에서 대화 조회
        conversations = memory_service.get_conversations(limit=1)
        assert len(conversations) == 1
        assert conversations[0].emotion == "happy"
        assert conversations[0].intent == "greeting"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
