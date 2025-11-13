# ====================================================================
#  File: services/cache.py
# ====================================================================

import json
import hashlib
import time
from pathlib import Path
from typing import Optional, Dict, Any


class ResponseCache:
    def __init__(
        self,
        cache_dir: str = "./cache",
        ttl: int = 3600,
        max_cache_size: int = 100,
        similarity_threshold: float = 0.9,
    ):
        """
        응답 캐싱 시스템
        
        Args:
            cache_dir (str): 캐시 파일 저장 디렉토리
            ttl (int): 캐시 유효 시간 (초)
            max_cache_size (int): 최대 캐시 항목 수
            similarity_threshold (float): 유사한 질문으로 판단할 임계값
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "response_cache.json"
        self.ttl = ttl
        self.max_cache_size = max_cache_size
        self.similarity_threshold = similarity_threshold
        
        self.stats = {
            "hits": 0,
            "misses": 0,
            "saves": 0
        }
    
    def _load_cache(self) -> Dict[str, Any]:
        """캐시 파일 로드"""
        if not self.cache_file.exists():
            return {}
        
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[L.U.N.A. Cache] 캐시 로드 실패: {e}")
            return {}
    
    def _save_cache(self, cache: Dict[str, Any]):
        """캐시 파일 저장"""
        try:
            if len(cache) > self.max_cache_size:
                sorted_items = sorted(
                    cache.items(),
                    key=lambda x: x[1].get("timestamp", 0)
                )
                cache = dict(sorted_items[-self.max_cache_size:])
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[L.U.N.A. Cache] 캐시 저장 실패: {e}")
    
    def _generate_key(self, prompt: str, model: str = "", context_hash: str = "") -> str:
        """
        프롬프트로부터 캐시 키 생성
        
        Args:
            prompt (str): 사용자 입력
            model (str): 모델 이름
            context_hash (str): 컨텍스트 해시 (대화 히스토리)
        
        Returns:
            str: 캐시 키 (SHA256 해시)
        """
        normalized = prompt.lower().strip()
        combined = f"{normalized}|{model}|{context_hash}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()
    
    def _is_expired(self, timestamp: float) -> bool:
        """캐시 만료 여부 확인"""
        return (time.time() - timestamp) > self.ttl
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        두 텍스트의 유사도 계산 (간단한 단어 기반)
        
        Returns:
            float: 0.0 ~ 1.0 사이의 유사도
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def get(
        self,
        prompt: str,
        model: str = "",
        context_hash: str = "",
        use_similarity: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        캐시에서 응답 조회
        
        Args:
            prompt (str): 사용자 입력
            model (str): 모델 이름
            context_hash (str): 컨텍스트 해시
            use_similarity (bool): 유사한 질문도 검색할지 여부
        
        Returns:
            Optional[Dict[str, Any]]: 캐시된 응답 또는 None
        """
        cache = self._load_cache()
        key = self._generate_key(prompt, model, context_hash)
        
        if key in cache:
            entry = cache[key]
            if not self._is_expired(entry["timestamp"]):
                self.stats["hits"] += 1
                print(f"[L.U.N.A. Cache] 캐시 히트 (정확 일치): {prompt[:50]}...")
                return entry["response"]
            else:
                del cache[key]
                self._save_cache(cache)
        
        if use_similarity:
            for cached_key, entry in cache.items():
                if self._is_expired(entry["timestamp"]):
                    continue
                
                cached_prompt = entry.get("prompt", "")
                similarity = self._calculate_similarity(prompt, cached_prompt)
                
                if similarity >= self.similarity_threshold:
                    self.stats["hits"] += 1
                    print(f"[L.U.N.A. Cache] 캐시 히트 (유사도: {similarity:.2f}): {cached_prompt[:50]}...")
                    return entry["response"]
        
        self.stats["misses"] += 1
        print(f"[L.U.N.A. Cache] 캐시 미스: {prompt[:50]}...")
        return None
    
    def set(
        self,
        prompt: str,
        response: Dict[str, Any],
        model: str = "",
        context_hash: str = ""
    ):
        """
        응답을 캐시에 저장
        
        Args:
            prompt (str): 사용자 입력
            response (Dict[str, Any]): LLM 응답
            model (str): 모델 이름
            context_hash (str): 컨텍스트 해시
        """
        cache = self._load_cache()
        key = self._generate_key(prompt, model, context_hash)
        
        cache[key] = {
            "prompt": prompt,
            "response": response,
            "model": model,
            "timestamp": time.time()
        }
        
        self._save_cache(cache)
        self.stats["saves"] += 1
        print(f"[L.U.N.A. Cache] 캐시 저장: {prompt[:50]}...")
    
    def clear(self):
        """모든 캐시 삭제"""
        try:
            if self.cache_file.exists():
                self.cache_file.unlink()
            print("[L.U.N.A. Cache] 캐시 삭제 완료")
        except Exception as e:
            print(f"[L.U.N.A. Cache] 캐시 삭제 실패: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        캐시 통계 반환
        
        Returns:
            Dict[str, Any]: 통계 정보
        """
        cache = self._load_cache()
        hit_rate = (
            self.stats["hits"] / (self.stats["hits"] + self.stats["misses"])
            if (self.stats["hits"] + self.stats["misses"]) > 0
            else 0.0
        )
        
        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "saves": self.stats["saves"],
            "hit_rate": f"{hit_rate * 100:.1f}%",
            "total_cached": len(cache),
            "max_size": self.max_cache_size,
            "ttl_seconds": self.ttl
        }
    
    def cleanup_expired(self):
        """만료된 캐시 항목 정리"""
        cache = self._load_cache()
        original_size = len(cache)
        
        cache = {
            k: v for k, v in cache.items()
            if not self._is_expired(v["timestamp"])
        }
        
        removed = original_size - len(cache)
        if removed > 0:
            self._save_cache(cache)
            print(f"[L.U.N.A. Cache] 만료된 캐시 {removed}개 삭제")
        
        return removed
