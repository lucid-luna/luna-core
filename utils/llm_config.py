# ====================================================================
#  File: utils/llm_config.py
# ====================================================================

import os
import yaml
from typing import Dict, Tuple, Optional
from services.llm_manager import LLMManager


def load_llm_config(config_path: str = "config/models.yaml") -> Dict:
    """
    YAML 설정 파일에서 LLM 설정을 로드합니다.
    
    Args:
        config_path (str): 설정 파일 경로
    
    Returns:
        Dict: 설정 딕셔너리
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"[L.U.N.A. Config] 설정 파일을 찾을 수 없습니다: {config_path}")
        return {}
    except Exception as e:
        print(f"[L.U.N.A. Config] 설정 파일 로드 중 오류: {e}")
        return {}


def create_llm_manager(
    mode: Optional[str] = None,
    config_path: str = "config/models.yaml"
) -> Tuple[Optional[LLMManager], Optional[str]]:
    """
    설정에 따라 LLM Manager를 생성합니다.
    
    Args:
        mode (str, optional): "server" 또는 "api". None이면 환경 변수나 설정 파일에서 읽음
        config_path (str): 설정 파일 경로
    
    Returns:
        Tuple[LLMManager, str]: (LLM Manager 인스턴스, 기본 타겟 이름)
    """
    config = load_llm_config(config_path)
    
    # LLM 설정이 있는지 확인
    llm_config = config.get("llm", {})
    
    # 모드 결정: 파라미터 > 환경 변수 > 설정 파일
    if mode is None:
        mode = os.getenv("LLM_MODE") or llm_config.get("mode", "server")
    
    if mode == "api":
        # API 모드 설정
        api_configs = llm_config.get("api", {})
        
        if not api_configs:
            # 기본 Gemini 설정
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GENAI_API_KEY")
            if api_key:
                api_configs = {
                    "gemini": {
                        "api_key": api_key,
                        "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
                    }
                }
            else:
                print("[L.U.N.A. Config] API 키가 설정되지 않았습니다.")
                return None, None
        
        llm = LLMManager(mode="api", api_configs=api_configs)
        default_target = list(api_configs.keys())[0]  # 첫 번째 API 제공자
        
        return llm, default_target
    
    elif mode == "server":
        # 서버 모드 설정
        server_configs = llm_config.get("servers", {})
        
        if not server_configs:
            # 기본 로컬 서버 설정
            server_configs = {
                "luna": {
                    "url": os.getenv("LLM_SERVER_URL", "http://localhost:8080"),
                    "alias": os.getenv("LLM_MODEL_ALIAS", "luna-model")
                }
            }
        
        llm = LLMManager(mode="server", server_configs=server_configs)
        default_target = list(server_configs.keys())[0]  # 첫 번째 서버
        
        return llm, default_target
    
    else:
        print(f"[L.U.N.A. Config] 지원하지 않는 모드입니다: {mode}")
        return None, None


def interactive_mode_selection() -> str:
    """
    사용자에게 모드를 선택하도록 합니다.
    
    Returns:
        str: "server" 또는 "api"
    """
    print("\n" + "="*50)
    print("L.U.N.A. LLM 모드 선택")
    print("="*50)
    print("1. 로컬 서버 모드 (Local LLM Server)")
    print("2. API 모드 (Gemini API 등)")
    print("="*50)
    
    while True:
        choice = input("모드를 선택하세요 (1 또는 2): ").strip()
        if choice == "1":
            return "server"
        elif choice == "2":
            return "api"
        else:
            print("올바른 선택이 아닙니다. 1 또는 2를 입력하세요.")


# 간편 사용 함수
def get_llm_manager(auto_mode: bool = False) -> Tuple[Optional[LLMManager], Optional[str]]:
    """
    LLM Manager를 자동 또는 대화형으로 가져옵니다.
    
    Args:
        auto_mode (bool): True면 환경 변수/설정 파일 기반, False면 사용자 선택
    
    Returns:
        Tuple[LLMManager, str]: (LLM Manager 인스턴스, 기본 타겟 이름)
    """
    if auto_mode:
        mode = os.getenv("LLM_MODE", "server")
    else:
        mode = interactive_mode_selection()
    
    return create_llm_manager(mode=mode)
