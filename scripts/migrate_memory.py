# ====================================================================
#  File: scripts/migrate_memory.py
# ====================================================================
"""
기존 JSON 메모리 데이터를 SQLite로 마이그레이션하는 스크립트
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.memory.database import get_db_manager
from services.memory.repository import MemoryRepository
from services.memory.models import ConversationCreate, SummaryCreate


def migrate_from_json(
    json_path: str = "./memory/conversation_history.json",
    db_path: str = "./memory/luna.db",
    user_id: str = "default",
    session_id: str = "default"
):
    """
    JSON 파일에서 SQLite로 데이터 마이그레이션
    
    Args:
        json_path: 기존 JSON 파일 경로
        db_path: 새 SQLite 파일 경로
        user_id: 사용자 ID
        session_id: 세션 ID
    """
    json_file = Path(json_path)
    
    if not json_file.exists():
        print(f"❌ JSON 파일이 없습니다: {json_path}")
        return
    
    print(f"📂 JSON 파일 로드 중: {json_path}")
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ JSON 로드 실패: {e}")
        return
    
    print(f"✅ {len(data)}개 항목 로드 완료")
    
    # DB 초기화
    print(f"📊 SQLite DB 초기화: {db_path}")
    db_manager = get_db_manager(db_path)
    repository = MemoryRepository(db_manager)
    
    # 데이터 변환 및 저장
    conversations_count = 0
    summaries_count = 0
    errors = []
    
    for i, entry in enumerate(data, 1):
        try:
            entry_type = entry.get("type", "conversation")
            
            if entry_type == "conversation":
                # 대화 항목
                timestamp_str = entry.get("timestamp", datetime.now().isoformat())
                
                # 타임스탬프 파싱 (여러 형식 지원)
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                except:
                    timestamp = datetime.now()
                
                # 메타데이터 추출
                metadata = entry.get("metadata", {})
                emotion = metadata.get("emotion")
                intent = metadata.get("intent")
                
                # 대화 생성
                conv = ConversationCreate(
                    user_id=user_id,
                    session_id=session_id,
                    user_message=entry.get("user", ""),
                    assistant_message=entry.get("assistant", ""),
                    emotion=emotion,
                    intent=intent,
                    metadata=metadata
                )
                
                repository.create_conversation(conv)
                conversations_count += 1
                
            elif entry_type == "summary":
                # 요약 항목
                timestamp_str = entry.get("timestamp", datetime.now().isoformat())
                
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                except:
                    timestamp = datetime.now()
                
                # 요약 생성
                summary = SummaryCreate(
                    user_id=user_id,
                    session_id=session_id,
                    content=entry.get("content", ""),
                    summarized_turns=entry.get("summarized_turns", 0)
                )
                
                repository.create_summary(summary)
                summaries_count += 1
            
            # 진행 상황 표시
            if i % 10 == 0:
                print(f"  진행 중: {i}/{len(data)} 항목 처리됨...")
        
        except Exception as e:
            error_msg = f"항목 {i} 처리 실패: {e}"
            errors.append(error_msg)
            print(f"⚠️  {error_msg}")
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("🎉 마이그레이션 완료!")
    print("=" * 60)
    print(f"✅ 대화: {conversations_count}개")
    print(f"✅ 요약: {summaries_count}개")
    
    if errors:
        print(f"⚠️  오류: {len(errors)}개")
        print("\n오류 목록:")
        for error in errors[:5]:  # 최대 5개만 표시
            print(f"  - {error}")
        if len(errors) > 5:
            print(f"  ... 외 {len(errors) - 5}개")
    
    # DB 정보 확인
    db_info = db_manager.get_db_info()
    print(f"\n📊 DB 정보:")
    print(f"  - 파일 크기: {db_info['file_size_mb']} MB")
    print(f"  - 대화 수: {db_info['conversations_count']}")
    print(f"  - 요약 수: {db_info['summaries_count']}")
    
    # 백업 권장
    backup_path = json_file.parent / f"{json_file.stem}_backup.json"
    print(f"\n💡 팁: 원본 JSON을 백업하세요!")
    print(f"   mv {json_file} {backup_path}")


def verify_migration(db_path: str = "./memory/luna.db"):
    """
    마이그레이션 결과 검증
    
    Args:
        db_path: SQLite 파일 경로
    """
    print("\n" + "=" * 60)
    print("🔍 마이그레이션 검증 중...")
    print("=" * 60)
    
    db_manager = get_db_manager(db_path)
    repository = MemoryRepository(db_manager)
    
    # 통계 조회
    stats = repository.get_stats()
    
    print(f"✅ 전체 대화: {stats.total_conversations}개")
    print(f"✅ 전체 요약: {stats.total_summaries}개")
    print(f"✅ 고유 사용자: {stats.unique_users}명")
    print(f"✅ 고유 세션: {stats.unique_sessions}개")
    
    if stats.first_conversation:
        print(f"📅 첫 대화: {stats.first_conversation.strftime('%Y-%m-%d %H:%M:%S')}")
    if stats.last_conversation:
        print(f"📅 마지막 대화: {stats.last_conversation.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 감정 분포
    if stats.emotions_distribution:
        print(f"\n😊 감정 분포:")
        for emotion, count in sorted(stats.emotions_distribution.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {emotion}: {count}개")
    
    # 의도 분포
    if stats.intents_distribution:
        print(f"\n🎯 의도 분포:")
        for intent, count in sorted(stats.intents_distribution.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {intent}: {count}개")
    
    # 최근 대화 5개 샘플
    print(f"\n📝 최근 대화 샘플 (5개):")
    recent = repository.get_conversations(limit=5)
    for i, conv in enumerate(recent, 1):
        user_preview = conv.user_message[:50] + "..." if len(conv.user_message) > 50 else conv.user_message
        print(f"  {i}. [{conv.timestamp.strftime('%m-%d %H:%M')}] {user_preview}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="JSON 메모리 데이터를 SQLite로 마이그레이션")
    parser.add_argument(
        "--json",
        default="./memory/conversation_history.json",
        help="JSON 파일 경로 (기본: ./memory/conversation_history.json)"
    )
    parser.add_argument(
        "--db",
        default="./memory/luna.db",
        help="SQLite 파일 경로 (기본: ./memory/luna.db)"
    )
    parser.add_argument(
        "--user-id",
        default="default",
        help="사용자 ID (기본: default)"
    )
    parser.add_argument(
        "--session-id",
        default="default",
        help="세션 ID (기본: default)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="마이그레이션 후 검증 실행"
    )
    
    args = parser.parse_args()
    
    # 마이그레이션 실행
    migrate_from_json(
        json_path=args.json,
        db_path=args.db,
        user_id=args.user_id,
        session_id=args.session_id
    )
    
    # 검증
    if args.verify:
        verify_migration(args.db)
    
    print("\n✨ 완료!")
