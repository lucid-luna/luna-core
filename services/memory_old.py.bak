# ====================================================================
#  File: services/memory.py
# ====================================================================

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


class MemoryService:
    def __init__(
        self,
        memory_dir: str = "./memory",
        max_entries: int = 50,
        max_context_turns: int = 6,
        summary_threshold: int = 20,
        enable_auto_summary: bool = True,
        llm_service=None
    ):
        """
        대화 메모리 관리 서비스
        
        Args:
            memory_dir (str): 메모리 파일 저장 디렉토리
            max_entries (int): 저장할 최대 대화 수 (오래된 것부터 삭제)
            max_context_turns (int): LLM에 전달할 최대 대화 턴 수
            summary_threshold (int): 자동 요약을 실행할 대화 턴 수 (기본: 20)
            enable_auto_summary (bool): 자동 요약 활성화 여부
            llm_service: LLM 서비스 인스턴스 (요약 생성용)
        """
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        self.max_entries = max_entries
        self.max_context_turns = max_context_turns
        self.summary_threshold = summary_threshold
        self.enable_auto_summary = enable_auto_summary
        self.llm_service = llm_service
        self.memory_file = self.memory_dir / "conversation_history.json"
        
        print(f"[MemoryService] 초기화 완료 (요약 임계값: {summary_threshold}턴, 자동 요약: {enable_auto_summary})")
        
    def load_memory(self) -> List[Dict[str, Any]]:
        """
        저장된 메모리를 로드합니다.
        
        Returns:
            List[Dict[str, Any]]: 대화 내역 리스트
        """
        if not self.memory_file.exists():
            return []
        
        try:
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[MemoryService] 메모리 로드 실패: {e}")
            return []
    
    def save_memory(self, memory: List[Dict[str, Any]]):
        """
        메모리를 파일에 저장합니다.
        
        Args:
            memory (List[Dict[str, Any]]): 저장할 대화 내역
        """
        try:
            # 최대 개수 제한
            if len(memory) > self.max_entries:
                memory = memory[-self.max_entries:]
            
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(memory, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[MemoryService] 메모리 저장 실패: {e}")
    
    def add_entry(self, user_input: str, assistant_response: str, metadata: Optional[Dict[str, Any]] = None):
        """
        새로운 대화를 메모리에 추가합니다.
        
        Args:
            user_input (str): 사용자 입력
            assistant_response (str): 어시스턴트 응답
            metadata (Dict, optional): 추가 메타데이터 (감정, 인텐트 등)
        """
        memory = self.load_memory()
        
        entry = {
            "type": "conversation",
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "assistant": assistant_response
        }
        
        if metadata:
            entry["metadata"] = metadata
        
        memory.append(entry)
        self.save_memory(memory)
        
        # 자동 요약 트리거
        if self.enable_auto_summary and self.llm_service:
            self._check_and_summarize()
    
    def get_context_for_llm(self, include_system_prompt: bool = False) -> List[Dict[str, str]]:
        """
        LLM에 전달할 대화 컨텍스트를 생성합니다.
        요약이 있으면 요약 + 최근 대화를 반환합니다.
        
        Args:
            include_system_prompt (bool): 시스템 프롬프트 포함 여부
        
        Returns:
            List[Dict[str, str]]: OpenAI 형식의 메시지 리스트
        """
        memory = self.load_memory()
        messages = []
        
        # 요약이 있는지 확인
        summary_entry = None
        conversations = []
        
        for entry in memory:
            if entry.get("type") == "summary":
                summary_entry = entry
            elif entry.get("type") == "conversation":
                conversations.append(entry)
        
        # 요약을 시스템 메시지로 추가
        if summary_entry:
            messages.append({
                "role": "system",
                "content": f"[이전 대화 요약]\n{summary_entry['content']}"
            })
        
        # 최근 N개의 대화만 추가
        recent_conversations = conversations[-self.max_context_turns:] if conversations else []
        
        for entry in recent_conversations:
            messages.append({"role": "user", "content": entry["user"]})
            messages.append({"role": "assistant", "content": entry["assistant"]})
        
        return messages
    
    def get_recent_summary(self, count: int = 5) -> str:
        """
        최근 대화 요약을 텍스트로 반환합니다.
        
        Args:
            count (int): 표시할 대화 수
        
        Returns:
            str: 대화 요약 텍스트
        """
        memory = self.load_memory()
        if not memory:
            return "(대화 내역 없음)"
        
        recent = memory[-count:]
        summary_parts = []
        
        for i, entry in enumerate(recent, 1):
            user_text = entry["user"][:50] + "..." if len(entry["user"]) > 50 else entry["user"]
            assistant_text = entry["assistant"][:50] + "..." if len(entry["assistant"]) > 50 else entry["assistant"]
            summary_parts.append(f"{i}. User: {user_text}\n   Assistant: {assistant_text}")
        
        return "\n\n".join(summary_parts)
    
    def clear_memory(self):
        """모든 메모리를 삭제합니다."""
        try:
            if self.memory_file.exists():
                self.memory_file.unlink()
            print("[MemoryService] 메모리 삭제 완료")
        except Exception as e:
            print(f"[MemoryService] 메모리 삭제 실패: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        메모리 통계 정보를 반환합니다.
        
        Returns:
            Dict[str, Any]: 메모리 통계
        """
        memory = self.load_memory()
        
        if not memory:
            return {
                "total_entries": 0,
                "first_conversation": None,
                "last_conversation": None
            }
        
        # 요약과 대화 분리
        conversations = [e for e in memory if e.get("type") == "conversation"]
        summaries = [e for e in memory if e.get("type") == "summary"]
        
        return {
            "total_entries": len(memory),
            "conversations": len(conversations),
            "summaries": len(summaries),
            "first_conversation": memory[0]["timestamp"] if memory else None,
            "last_conversation": memory[-1]["timestamp"] if memory else None,
            "context_window": self.max_context_turns,
            "max_stored": self.max_entries,
            "auto_summary_enabled": self.enable_auto_summary,
            "summary_threshold": self.summary_threshold
        }
    
    def _check_and_summarize(self):
        """
        대화 수를 확인하고 임계값을 초과하면 자동으로 요약합니다.
        """
        memory = self.load_memory()
        conversations = [e for e in memory if e.get("type") == "conversation"]
        
        if len(conversations) >= self.summary_threshold:
            print(f"[MemoryService] 대화 {len(conversations)}턴 도달 → 자동 요약 실행")
            self._summarize_conversations()
    
    def _summarize_conversations(self):
        """
        오래된 대화를 요약하고 메모리를 재구성합니다.
        """
        memory = self.load_memory()
        
        # 요약과 대화 분리
        existing_summary = None
        conversations = []
        
        for entry in memory:
            if entry.get("type") == "summary":
                existing_summary = entry
            elif entry.get("type") == "conversation":
                conversations.append(entry)
        
        if len(conversations) <= self.max_context_turns:
            print("[MemoryService] 대화가 충분하지 않아 요약을 건너뜁니다.")
            return
        
        # 최근 대화는 유지, 오래된 대화만 요약
        recent_conversations = conversations[-self.max_context_turns:]
        old_conversations = conversations[:-self.max_context_turns]
        
        if not old_conversations:
            return
        
        # 요약 생성
        new_summary = self._generate_summary(old_conversations, existing_summary)
        
        if new_summary:
            # 메모리 재구성: [새 요약] + [최근 대화]
            new_memory = []
            
            # 요약 추가
            summary_entry = {
                "type": "summary",
                "content": new_summary,
                "summarized_turns": len(old_conversations),
                "timestamp": datetime.now().isoformat()
            }
            new_memory.append(summary_entry)
            
            # 최근 대화 추가
            new_memory.extend(recent_conversations)
            
            # 저장
            self.save_memory(new_memory)
            print(f"[MemoryService] 요약 완료 ({len(old_conversations)}턴 → 요약)")
    
    def _generate_summary(
        self,
        conversations: List[Dict[str, Any]],
        existing_summary: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        LLM을 사용하여 대화를 요약합니다.
        
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
            conversation_text.append(f"User: {conv['user']}")
            conversation_text.append(f"Assistant: {conv['assistant']}")
        
        conversation_str = "\n".join(conversation_text)
        
        # 요약 프롬프트 생성
        if existing_summary:
            prompt = f"""기존 요약:
{existing_summary['content']}

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
            
            # 응답에서 텍스트 추출
            if isinstance(response, dict) and "choices" in response:
                summary_text = response["choices"][0]["message"]["content"]
                return summary_text.strip()
            
            return None
        
        except Exception as e:
            print(f"[MemoryService] 요약 생성 중 오류 발생: {e}")
            return None
    
    def force_summarize(self) -> bool:
        """
        수동으로 요약을 강제 실행합니다.
        
        Returns:
            bool: 요약 성공 여부
        """
        print("[MemoryService] 수동 요약 실행")
        self._summarize_conversations()
        return True
    
    def get_summary(self) -> Optional[str]:
        """
        현재 저장된 요약을 반환합니다.
        
        Returns:
            str: 요약 텍스트 (없으면 None)
        """
        memory = self.load_memory()
        
        for entry in memory:
            if entry.get("type") == "summary":
                return entry.get("content")
        
        return None
