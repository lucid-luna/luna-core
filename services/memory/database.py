# ====================================================================
#  File: services/memory/database.py
# ====================================================================
"""
SQLite 데이터베이스 스키마 및 연결 관리
"""

import sqlite3
from pathlib import Path
from typing import Optional
from contextlib import contextmanager
import threading


class DatabaseManager:
    """SQLite 데이터베이스 관리자"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, db_path: str = "./memory/luna.db"):
        """싱글톤 패턴으로 DB 연결 관리"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, db_path: str = "./memory/luna.db"):
        """
        데이터베이스 초기화
        
        Args:
            db_path: SQLite 파일 경로
        """
        if self._initialized:
            return
            
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 연결 풀 (스레드별 연결)
        self._local = threading.local()
        
        # 스키마 초기화
        self._init_schema()
        self._initialized = True
        
        print(f"[DatabaseManager] SQLite 초기화 완료: {self.db_path}")
    
    def get_connection(self) -> sqlite3.Connection:
        """
        현재 스레드의 데이터베이스 연결 반환
        
        Returns:
            sqlite3.Connection: 데이터베이스 연결
        """
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0
            )
            # Row factory 설정 (딕셔너리 형태로 반환)
            self._local.connection.row_factory = sqlite3.Row
            # Foreign key 활성화
            self._local.connection.execute("PRAGMA foreign_keys = ON")
            
        return self._local.connection
    
    @contextmanager
    def get_cursor(self):
        """
        컨텍스트 매니저로 커서 제공 (자동 커밋/롤백)
        
        Yields:
            sqlite3.Cursor: 데이터베이스 커서
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
    
    def _init_schema(self):
        """데이터베이스 스키마 초기화"""
        with self.get_cursor() as cursor:
            # conversations 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL DEFAULT 'default',
                    session_id TEXT NOT NULL DEFAULT 'default',
                    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    user_message TEXT NOT NULL,
                    assistant_message TEXT NOT NULL,
                    emotion TEXT,
                    intent TEXT,
                    processing_time REAL,
                    cached INTEGER DEFAULT 0,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # summaries 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL DEFAULT 'default',
                    session_id TEXT NOT NULL DEFAULT 'default',
                    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    content TEXT NOT NULL,
                    summarized_turns INTEGER NOT NULL,
                    start_conversation_id INTEGER,
                    end_conversation_id INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (start_conversation_id) REFERENCES conversations(id),
                    FOREIGN KEY (end_conversation_id) REFERENCES conversations(id)
                )
            """)
            
            # 인덱스 생성
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_user_session 
                ON conversations(user_id, session_id, timestamp DESC)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_timestamp 
                ON conversations(timestamp DESC)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_emotion 
                ON conversations(emotion)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_intent 
                ON conversations(intent)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_summaries_user_session 
                ON summaries(user_id, session_id, timestamp DESC)
            """)
            
            # FTS (Full-Text Search) 테이블 생성 (검색 최적화)
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS conversations_fts 
                USING fts5(
                    user_message, 
                    assistant_message,
                    content='conversations',
                    content_rowid='id'
                )
            """)
            
            # FTS 트리거 설정 (자동 인덱싱)
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS conversations_fts_insert 
                AFTER INSERT ON conversations 
                BEGIN
                    INSERT INTO conversations_fts(rowid, user_message, assistant_message)
                    VALUES (new.id, new.user_message, new.assistant_message);
                END
            """)
            
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS conversations_fts_delete 
                AFTER DELETE ON conversations 
                BEGIN
                    DELETE FROM conversations_fts WHERE rowid = old.id;
                END
            """)
            
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS conversations_fts_update 
                AFTER UPDATE ON conversations 
                BEGIN
                    DELETE FROM conversations_fts WHERE rowid = old.id;
                    INSERT INTO conversations_fts(rowid, user_message, assistant_message)
                    VALUES (new.id, new.user_message, new.assistant_message);
                END
            """)
    
    def close(self):
        """데이터베이스 연결 종료"""
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None
    
    def vacuum(self):
        """데이터베이스 최적화 (공간 회수)"""
        with self.get_cursor() as cursor:
            cursor.execute("VACUUM")
        print("[DatabaseManager] 데이터베이스 최적화 완료")
    
    def get_db_info(self) -> dict:
        """
        데이터베이스 정보 반환
        
        Returns:
            dict: DB 정보 (크기, 테이블 수 등)
        """
        with self.get_cursor() as cursor:
            # 파일 크기
            file_size = self.db_path.stat().st_size if self.db_path.exists() else 0
            
            # 테이블별 레코드 수
            cursor.execute("SELECT COUNT(*) as count FROM conversations")
            conversations_count = cursor.fetchone()['count']
            
            cursor.execute("SELECT COUNT(*) as count FROM summaries")
            summaries_count = cursor.fetchone()['count']
            
            return {
                "db_path": str(self.db_path),
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size / 1024 / 1024, 2),
                "conversations_count": conversations_count,
                "summaries_count": summaries_count
            }


# 전역 인스턴스
_db_manager: Optional[DatabaseManager] = None


def get_db_manager(db_path: str = "./memory/luna.db") -> DatabaseManager:
    """
    DatabaseManager 싱글톤 인스턴스 반환
    
    Args:
        db_path: SQLite 파일 경로
        
    Returns:
        DatabaseManager: 데이터베이스 관리자
    """
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager(db_path)
    return _db_manager
