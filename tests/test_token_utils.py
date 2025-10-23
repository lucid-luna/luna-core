# ====================================================================
#  File: tests/test_token_utils.py
# ====================================================================
"""
토큰 유틸리티 테스트
"""

import pytest
from services.memory.token_utils import (
    estimate_tokens_for_text,
    estimate_tokens_for_messages,
    CHARS_PER_TOKEN
)


class TestEstimateTokensForText:
    """텍스트 토큰 추정 테스트"""
    
    def test_empty_text(self):
        """빈 텍스트는 0 토큰"""
        assert estimate_tokens_for_text("") == 0
        assert estimate_tokens_for_text(None) == 0
    
    def test_short_text(self):
        """짧은 텍스트 토큰 수 확인"""
        text = "안녕하세요"  # 5자
        tokens = estimate_tokens_for_text(text)
        # 5 / 4.0 = 1.25 -> 1 토큰 (최소 1)
        assert tokens >= 1
        assert tokens <= 5  # 보수적으로 추정
    
    def test_medium_text(self):
        """중간 길이 텍스트"""
        text = "안녕하세요. 오늘 날씨가 참 좋네요. 산책하기 딱 좋은 날씨입니다."  # 약 34자
        tokens = estimate_tokens_for_text(text)
        # 34 / 4.0 = 8.5 -> 9 토큰
        expected = int(round(len(text) / CHARS_PER_TOKEN))
        assert tokens == expected
        assert tokens >= 5
        assert tokens <= 20
    
    def test_long_text(self):
        """긴 텍스트"""
        text = "Lorem ipsum dolor sit amet, " * 50  # 약 1400자
        tokens = estimate_tokens_for_text(text)
        # 대략 300-400 토큰 예상
        assert tokens >= 200
        assert tokens <= 500
    
    def test_mixed_language(self):
        """한글/영어 혼합 텍스트"""
        text = "Hello 안녕하세요 World 세계"
        tokens = estimate_tokens_for_text(text)
        assert tokens >= 3
        assert tokens <= 10


class TestEstimateTokensForMessages:
    """메시지 리스트 토큰 추정 테스트"""
    
    def test_empty_messages(self):
        """빈 메시지 리스트"""
        assert estimate_tokens_for_messages([]) == 0
        assert estimate_tokens_for_messages(None) == 0
    
    def test_single_message(self):
        """단일 메시지"""
        messages = [
            {"role": "user", "content": "안녕하세요"}
        ]
        tokens = estimate_tokens_for_messages(messages)
        # 텍스트 토큰 + 오버헤드(4)
        text_tokens = estimate_tokens_for_text("안녕하세요")
        assert tokens == text_tokens + 4
    
    def test_multiple_messages(self):
        """여러 메시지"""
        messages = [
            {"role": "user", "content": "안녕하세요"},
            {"role": "assistant", "content": "네, 안녕하세요!"},
            {"role": "user", "content": "날씨가 좋네요"}
        ]
        tokens = estimate_tokens_for_messages(messages)
        
        # 각 메시지 토큰 + 오버헤드
        expected = 0
        for msg in messages:
            expected += estimate_tokens_for_text(msg["content"]) + 4
        
        assert tokens == expected
        assert tokens > 0
    
    def test_long_conversation(self):
        """긴 대화"""
        messages = []
        for i in range(20):
            messages.append({"role": "user", "content": f"질문 {i}: 이것은 테스트 메시지입니다."})
            messages.append({"role": "assistant", "content": f"답변 {i}: 네, 알겠습니다."})
        
        tokens = estimate_tokens_for_messages(messages)
        # 40개 메시지 * (텍스트 토큰 + 오버헤드 4) 
        # 대략 400-800 토큰 예상
        assert tokens >= 300
        assert tokens <= 1000
    
    def test_message_with_empty_content(self):
        """빈 content 메시지"""
        messages = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "응답"}
        ]
        tokens = estimate_tokens_for_messages(messages)
        # 첫 메시지는 오버헤드만, 두번째는 텍스트 + 오버헤드
        assert tokens >= 8  # 최소 오버헤드 * 2
    
    def test_system_message(self):
        """시스템 메시지 포함"""
        messages = [
            {"role": "system", "content": "당신은 도움이 되는 AI 어시스턴트입니다."},
            {"role": "user", "content": "안녕하세요"},
            {"role": "assistant", "content": "네, 안녕하세요!"}
        ]
        tokens = estimate_tokens_for_messages(messages)
        assert tokens > 0
        
        # 각 메시지 개별 계산과 비교
        expected = sum(
            estimate_tokens_for_text(msg["content"]) + 4
            for msg in messages
        )
        assert tokens == expected


class TestTokenEstimationConsistency:
    """토큰 추정 일관성 테스트"""
    
    def test_same_text_same_tokens(self):
        """같은 텍스트는 항상 같은 토큰 수"""
        text = "일관성 테스트 텍스트"
        tokens1 = estimate_tokens_for_text(text)
        tokens2 = estimate_tokens_for_text(text)
        assert tokens1 == tokens2
    
    def test_longer_text_more_tokens(self):
        """더 긴 텍스트는 더 많은 토큰"""
        short = "짧음"
        long = "이것은 훨씬 더 긴 텍스트입니다. 많은 단어와 문장이 포함되어 있습니다."
        
        short_tokens = estimate_tokens_for_text(short)
        long_tokens = estimate_tokens_for_text(long)
        
        assert long_tokens > short_tokens
    
    def test_message_count_affects_overhead(self):
        """메시지 개수가 많을수록 오버헤드 증가"""
        # 같은 내용을 1개 메시지 vs 10개 메시지로
        content = "테스트"
        
        one_message = [{"role": "user", "content": content * 10}]
        ten_messages = [{"role": "user", "content": content} for _ in range(10)]
        
        tokens_one = estimate_tokens_for_messages(one_message)
        tokens_ten = estimate_tokens_for_messages(ten_messages)
        
        # 10개 메시지는 오버헤드가 10배
        # (오버헤드 = 4 * 메시지 수)
        overhead_diff = 4 * 9  # 9개 메시지 차이
        assert tokens_ten >= tokens_one + overhead_diff


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
