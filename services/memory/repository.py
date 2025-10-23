# ====================================================================
#  File: services/memory/repository.py
# ====================================================================
"""
메모리 데이터베이스 Repository (CRUD + 검색)
"""

import json
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from .database import DatabaseManager
from .models import (
    ConversationCreate,
    ConversationResponse,
    SummaryCreate,
    SummaryResponse,
    ConversationSearchRequest,
    MemoryStats
)


class MemoryRepository:
    """메모리 데이터 접근 레이어"""
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Repository 초기화
        
        Args:
            db_manager: 데이터베이스 관리자
        """
        self.db = db_manager
    
    # ==================== Conversation CRUD ====================
    
    def create_conversation(self, conv: ConversationCreate) -> ConversationResponse:
        """
        대화 생성
        
        Args:
            conv: 대화 생성 데이터
            
        Returns:
            ConversationResponse: 생성된 대화
        """
        with self.db.get_cursor() as cursor:
            metadata_json = json.dumps(conv.metadata) if conv.metadata else None
            
            cursor.execute("""
                INSERT INTO conversations 
                (user_id, session_id, user_message, assistant_message, 
                 emotion, intent, processing_time, cached, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                conv.user_id,
                conv.session_id,
                conv.user_message,
                conv.assistant_message,
                conv.emotion,
                conv.intent,
                conv.processing_time,
                1 if conv.cached else 0,
                metadata_json
            ))
            
            conv_id = cursor.lastrowid
            return self.get_conversation_by_id(conv_id)
    
    def get_conversation_by_id(self, conv_id: int) -> Optional[ConversationResponse]:
        """
        ID로 대화 조회
        
        Args:
            conv_id: 대화 ID
            
        Returns:
            ConversationResponse: 대화 데이터 (없으면 None)
        """
        with self.db.get_cursor() as cursor:
            cursor.execute("""
                SELECT * FROM conversations WHERE id = ?
            """, (conv_id,))
            
            row = cursor.fetchone()
            if row:
                return self._row_to_conversation(row)
            return None
    
    def get_conversations(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[ConversationResponse]:
        """
        대화 목록 조회 (페이지네이션)
        
        Args:
            user_id: 사용자 ID (None이면 모두)
            session_id: 세션 ID (None이면 모두)
            limit: 최대 개수
            offset: 시작 위치
            
        Returns:
            List[ConversationResponse]: 대화 목록
        """
        with self.db.get_cursor() as cursor:
            query = "SELECT * FROM conversations WHERE 1=1"
            params = []
            
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            
            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)
            
            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [self._row_to_conversation(row) for row in rows]
    
    def search_conversations(self, search: ConversationSearchRequest) -> Tuple[List[ConversationResponse], int]:
        """
        대화 검색 (키워드, 감정, 의도, 날짜 등)
        
        Args:
            search: 검색 조건
            
        Returns:
            Tuple[List[ConversationResponse], int]: (검색 결과, 전체 개수)
        """
        with self.db.get_cursor() as cursor:
            # 기본 쿼리
            query = "SELECT * FROM conversations WHERE 1=1"
            count_query = "SELECT COUNT(*) as count FROM conversations WHERE 1=1"
            params = []
            
            # 조건 추가
            if search.user_id:
                query += " AND user_id = ?"
                count_query += " AND user_id = ?"
                params.append(search.user_id)
            
            if search.session_id:
                query += " AND session_id = ?"
                count_query += " AND session_id = ?"
                params.append(search.session_id)
            
            if search.emotion:
                query += " AND emotion = ?"
                count_query += " AND emotion = ?"
                params.append(search.emotion)
            
            if search.intent:
                query += " AND intent = ?"
                count_query += " AND intent = ?"
                params.append(search.intent)
            
            if search.date_from:
                query += " AND timestamp >= ?"
                count_query += " AND timestamp >= ?"
                params.append(search.date_from.isoformat())
            
            if search.date_to:
                query += " AND timestamp <= ?"
                count_query += " AND timestamp <= ?"
                params.append(search.date_to.isoformat())
            
            # 키워드 검색 (FTS 사용)
            if search.keyword:
                query = """
                    SELECT c.* FROM conversations c
                    JOIN conversations_fts fts ON c.id = fts.rowid
                    WHERE fts MATCH ?
                """
                count_query = """
                    SELECT COUNT(*) as count FROM conversations c
                    JOIN conversations_fts fts ON c.id = fts.rowid
                    WHERE fts MATCH ?
                """
                # FTS 쿼리로 파라미터 재구성
                params = [search.keyword] + params[:-2] if len(params) > 2 else [search.keyword]
            
            # 전체 개수 조회
            cursor.execute(count_query, params)
            total_count = cursor.fetchone()['count']
            
            # 정렬 및 페이지네이션
            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([search.limit, search.offset])
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            results = [self._row_to_conversation(row) for row in rows]
            return results, total_count
    
    def delete_conversation(self, conv_id: int) -> bool:
        """
        대화 삭제
        
        Args:
            conv_id: 대화 ID
            
        Returns:
            bool: 삭제 성공 여부
        """
        with self.db.get_cursor() as cursor:
            cursor.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
            return cursor.rowcount > 0
    
    def delete_conversations(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        before_date: Optional[datetime] = None
    ) -> int:
        """
        조건에 맞는 대화 일괄 삭제
        
        Args:
            user_id: 사용자 ID
            session_id: 세션 ID
            before_date: 기준 날짜 (이전 것들 삭제)
            
        Returns:
            int: 삭제된 개수
        """
        with self.db.get_cursor() as cursor:
            query = "DELETE FROM conversations WHERE 1=1"
            params = []
            
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            
            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)
            
            if before_date:
                query += " AND timestamp < ?"
                params.append(before_date.isoformat())
            
            cursor.execute(query, params)
            return cursor.rowcount
    
    # ==================== Summary CRUD ====================
    
    def create_summary(self, summary: SummaryCreate) -> SummaryResponse:
        """
        요약 생성
        
        Args:
            summary: 요약 생성 데이터
            
        Returns:
            SummaryResponse: 생성된 요약
        """
        with self.db.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO summaries 
                (user_id, session_id, content, summarized_turns, 
                 start_conversation_id, end_conversation_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                summary.user_id,
                summary.session_id,
                summary.content,
                summary.summarized_turns,
                summary.start_conversation_id,
                summary.end_conversation_id
            ))
            
            summary_id = cursor.lastrowid
            return self.get_summary_by_id(summary_id)
    
    def get_summary_by_id(self, summary_id: int) -> Optional[SummaryResponse]:
        """
        ID로 요약 조회
        
        Args:
            summary_id: 요약 ID
            
        Returns:
            SummaryResponse: 요약 데이터 (없으면 None)
        """
        with self.db.get_cursor() as cursor:
            cursor.execute("""
                SELECT * FROM summaries WHERE id = ?
            """, (summary_id,))
            
            row = cursor.fetchone()
            if row:
                return self._row_to_summary(row)
            return None
    
    def get_summaries(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 10
    ) -> List[SummaryResponse]:
        """
        요약 목록 조회
        
        Args:
            user_id: 사용자 ID
            session_id: 세션 ID
            limit: 최대 개수
            
        Returns:
            List[SummaryResponse]: 요약 목록
        """
        with self.db.get_cursor() as cursor:
            query = "SELECT * FROM summaries WHERE 1=1"
            params = []
            
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            
            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [self._row_to_summary(row) for row in rows]
    
    def get_latest_summary(
        self,
        user_id: str = "default",
        session_id: str = "default"
    ) -> Optional[SummaryResponse]:
        """
        최신 요약 조회
        
        Args:
            user_id: 사용자 ID
            session_id: 세션 ID
            
        Returns:
            SummaryResponse: 최신 요약 (없으면 None)
        """
        summaries = self.get_summaries(user_id, session_id, limit=1)
        return summaries[0] if summaries else None
    
    def delete_summary(self, summary_id: int) -> bool:
        """
        요약 삭제
        
        Args:
            summary_id: 요약 ID
            
        Returns:
            bool: 삭제 성공 여부
        """
        with self.db.get_cursor() as cursor:
            cursor.execute("DELETE FROM summaries WHERE id = ?", (summary_id,))
            return cursor.rowcount > 0
    
    # ==================== 통계 ====================
    
    def get_stats(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> MemoryStats:
        """
        메모리 통계 조회
        
        Args:
            user_id: 사용자 ID
            session_id: 세션 ID
            
        Returns:
            MemoryStats: 통계 데이터
        """
        with self.db.get_cursor() as cursor:
            # 기본 쿼리 조건
            where_clause = "WHERE 1=1"
            params = []
            
            if user_id:
                where_clause += " AND user_id = ?"
                params.append(user_id)
            
            if session_id:
                where_clause += " AND session_id = ?"
                params.append(session_id)
            
            # 전체 대화 수
            cursor.execute(f"SELECT COUNT(*) as count FROM conversations {where_clause}", params)
            total_conversations = cursor.fetchone()['count']
            
            # 전체 요약 수
            cursor.execute(f"SELECT COUNT(*) as count FROM summaries {where_clause}", params)
            total_summaries = cursor.fetchone()['count']
            
            # 고유 사용자 수
            cursor.execute("SELECT COUNT(DISTINCT user_id) as count FROM conversations")
            unique_users = cursor.fetchone()['count']
            
            # 고유 세션 수
            cursor.execute("SELECT COUNT(DISTINCT session_id) as count FROM conversations")
            unique_sessions = cursor.fetchone()['count']
            
            # 첫/마지막 대화
            cursor.execute(f"""
                SELECT MIN(timestamp) as first, MAX(timestamp) as last 
                FROM conversations {where_clause}
            """, params)
            row = cursor.fetchone()
            first_conv = datetime.fromisoformat(row['first']) if row['first'] else None
            last_conv = datetime.fromisoformat(row['last']) if row['last'] else None
            
            # 감정 분포
            cursor.execute(f"""
                SELECT emotion, COUNT(*) as count 
                FROM conversations {where_clause} AND emotion IS NOT NULL
                GROUP BY emotion
            """, params)
            emotions_dist = {row['emotion']: row['count'] for row in cursor.fetchall()}
            
            # 의도 분포
            cursor.execute(f"""
                SELECT intent, COUNT(*) as count 
                FROM conversations {where_clause} AND intent IS NOT NULL
                GROUP BY intent
            """, params)
            intents_dist = {row['intent']: row['count'] for row in cursor.fetchall()}
            
            # 평균 처리 시간
            cursor.execute(f"""
                SELECT AVG(processing_time) as avg_time 
                FROM conversations {where_clause} AND processing_time IS NOT NULL
            """, params)
            avg_time = cursor.fetchone()['avg_time']
            
            # 캐시 히트율
            cursor.execute(f"""
                SELECT 
                    SUM(CASE WHEN cached = 1 THEN 1 ELSE 0 END) as hits,
                    COUNT(*) as total
                FROM conversations {where_clause}
            """, params)
            row = cursor.fetchone()
            cache_hit_rate = (row['hits'] / row['total'] * 100) if row['total'] > 0 else 0
            
            # 날짜별 대화 수
            cursor.execute(f"""
                SELECT DATE(timestamp) as date, COUNT(*) as count
                FROM conversations {where_clause}
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
                LIMIT 30
            """, params)
            conversations_by_date = {row['date']: row['count'] for row in cursor.fetchall()}
            
            return MemoryStats(
                total_conversations=total_conversations,
                total_summaries=total_summaries,
                unique_users=unique_users,
                unique_sessions=unique_sessions,
                first_conversation=first_conv,
                last_conversation=last_conv,
                emotions_distribution=emotions_dist,
                intents_distribution=intents_dist,
                avg_processing_time=avg_time,
                cache_hit_rate=cache_hit_rate,
                conversations_by_date=conversations_by_date
            )
    
    # ==================== Helper Methods ====================
    
    def _row_to_conversation(self, row) -> ConversationResponse:
        """Row를 ConversationResponse로 변환"""
        metadata = json.loads(row['metadata']) if row['metadata'] else None
        
        return ConversationResponse(
            id=row['id'],
            user_id=row['user_id'],
            session_id=row['session_id'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            user_message=row['user_message'],
            assistant_message=row['assistant_message'],
            emotion=row['emotion'],
            intent=row['intent'],
            processing_time=row['processing_time'],
            cached=bool(row['cached']),
            metadata=metadata,
            created_at=datetime.fromisoformat(row['created_at'])
        )
    
    def _row_to_summary(self, row) -> SummaryResponse:
        """Row를 SummaryResponse로 변환"""
        return SummaryResponse(
            id=row['id'],
            user_id=row['user_id'],
            session_id=row['session_id'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            content=row['content'],
            summarized_turns=row['summarized_turns'],
            start_conversation_id=row['start_conversation_id'],
            end_conversation_id=row['end_conversation_id'],
            created_at=datetime.fromisoformat(row['created_at'])
        )
