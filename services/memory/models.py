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
    emotions_distribution: Dict[str, int] = Field(default_factory=dict)
    intents_distribution: Dict[str, int] = Field(default_factory=dict)
    avg_processing_time: Optional[float] = None
    cache_hit_rate: Optional[float] = None
    conversations_by_date: Dict[str, int] = Field(default_factory=dict)


class LLMContext(BaseModel):
    """LLM 컨텍스트 모델"""
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


# ====================================================================
#  장기/단기 메모리 모델
# ====================================================================

class MemoryType(str):
    """메모리 타입 상수"""
    CORE = "core"           # 장기 메모리 (영구 저장)
    WORKING = "working"     # 단기 메모리 (만료됨)


class CoreMemoryCategory(str):
    """장기 메모리 카테고리"""
    USER_INFO = "user_info"         # 사용자 정보 (이름, 성격 등)
    PREFERENCES = "preferences"     # 선호도 (좋아하는 것, 싫어하는 것)
    PROJECTS = "projects"           # 프로젝트 정보
    RELATIONSHIPS = "relationships" # 관계 정보 (호칭, 대화 스타일)
    FACTS = "facts"                 # 중요한 사실


class CoreMemoryCreate(BaseModel):
    """장기 메모리 생성 요청"""
    user_id: str = "default"
    category: str  # CoreMemoryCategory 값
    key: str       # 고유 키 (예: "name", "project_luna")
    value: str     # 저장할 내용
    importance: int = Field(default=5, ge=1, le=10)  # 중요도 1-10
    source: Optional[str] = None  # 출처 (대화 ID 등)
    metadata: Optional[Dict[str, Any]] = None


class CoreMemoryResponse(BaseModel):
    """장기 메모리 응답"""
    id: int
    user_id: str
    category: str
    key: str
    value: str
    importance: int
    source: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class WorkingMemoryCreate(BaseModel):
    """단기 메모리 생성 요청"""
    user_id: str = "default"
    session_id: str = "default"
    topic: str          # 주제 (예: "노션 페이지 작업")
    content: str        # 내용
    expires_at: Optional[datetime] = None  # 만료 시간 (기본: 3일 후)
    importance: int = Field(default=3, ge=1, le=10)
    metadata: Optional[Dict[str, Any]] = None


class WorkingMemoryResponse(BaseModel):
    """단기 메모리 응답"""
    id: int
    user_id: str
    session_id: str
    topic: str
    content: str
    importance: int
    expires_at: datetime
    is_expired: bool = False
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime

    class Config:
        from_attributes = True


class MemoryExtraction(BaseModel):
    """LLM이 추출한 메모리 정보"""
    core_memories: List[CoreMemoryCreate] = Field(default_factory=list)
    working_memories: List[WorkingMemoryCreate] = Field(default_factory=list)
    should_update: List[str] = Field(default_factory=list)  # 업데이트할 기존 메모리 키들