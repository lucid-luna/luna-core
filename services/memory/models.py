# ====================================================================
#  File: services/memory/models.py
# ====================================================================
"""
메모리 시스템 데이터 모델
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class ConversationCreate(BaseModel):
    """대화 생성 요청 모델"""
    user_id: str = "default"
    session_id: str = "default"
    user_message: str
    assistant_message: str
    emotion: Optional[str] = None
    intent: Optional[str] = None
    processing_time: Optional[float] = None
    cached: bool = False
    metadata: Optional[Dict[str, Any]] = None


class ConversationResponse(BaseModel):
    """대화 응답 모델"""
    id: int
    user_id: str
    session_id: str
    timestamp: datetime
    user_message: str
    assistant_message: str
    emotion: Optional[str] = None
    intent: Optional[str] = None
    processing_time: Optional[float] = None
    cached: bool = False
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class SummaryCreate(BaseModel):
    """요약 생성 요청 모델"""
    user_id: str = "default"
    session_id: str = "default"
    content: str
    summarized_turns: int
    start_conversation_id: Optional[int] = None
    end_conversation_id: Optional[int] = None


class SummaryResponse(BaseModel):
    """요약 응답 모델"""
    id: int
    user_id: str
    session_id: str
    timestamp: datetime
    content: str
    summarized_turns: int
    start_conversation_id: Optional[int] = None
    end_conversation_id: Optional[int] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class ConversationSearchRequest(BaseModel):
    """대화 검색 요청 모델"""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    keyword: Optional[str] = None
    emotion: Optional[str] = None
    intent: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    limit: int = Field(default=50, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)


class MemoryStats(BaseModel):
    """메모리 통계 모델"""
    total_conversations: int
    total_summaries: int
    unique_users: int
    unique_sessions: int
    first_conversation: Optional[datetime] = None
    last_conversation: Optional[datetime] = None
    emotions_distribution: Dict[str, int] = {}
    intents_distribution: Dict[str, int] = {}
    avg_processing_time: Optional[float] = None
    cache_hit_rate: Optional[float] = None
    conversations_by_date: Dict[str, int] = {}


class LLMContext(BaseModel):
    """LLM 컨텍스트 모델 (OpenAI 형식)"""
    role: str  # "system", "user", "assistant"
    content: str


class MemoryCleanupRequest(BaseModel):
    """메모리 정리 요청 모델"""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    before_date: Optional[datetime] = None
    keep_summaries: bool = True


class MemoryCleanupResponse(BaseModel):
    """메모리 정리 응답 모델"""
    deleted_conversations: int
    deleted_summaries: int
    message: str
