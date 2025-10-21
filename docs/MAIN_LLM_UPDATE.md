# L.U.N.A. LLM 통합 업데이트 가이드

## 변경 사항 요약

`main.py`가 이제 `LLMManager`를 사용하여 로컬 LLM 서버와 Gemini API를 선택적으로 사용할 수 있습니다.

## 주요 변경 파일

### 1. `main.py`
- `LLMService` → `LLMManager` 사용
- 설정 파일(`config/models.yaml`)에서 자동으로 LLM 모드 로드
- 폴백: 설정 로드 실패 시 기본 서버 모드로 대체

### 2. `services/interaction.py`
- `LLMService` → `LLMManager` 타입 변경
- `llm_target` 파라미터 추가 (타겟 서버/API 지정)
- `target_server="rp"` → `target=self.llm_target` 로 변경

### 3. `config/models.yaml`
- LLM 설정 섹션 추가
- 서버 이름 `rp`, `translator`로 통일

## 사용 방법

### 방법 1: 로컬 서버 모드 (기본값)

`config/models.yaml` 설정:
```yaml
llm:
  mode: "server"
  servers:
    rp:
      url: "http://localhost:8080"
      alias: "Luna"
```

서버 실행:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 방법 2: Gemini API 모드

환경 변수 설정:
```powershell
# Windows PowerShell
$env:GEMINI_API_KEY="your-api-key-here"
```

`config/models.yaml` 설정:
```yaml
llm:
  mode: "api"
  api:
    gemini:
      model: "gemini-2.0-flash-exp"
```

서버 실행:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 방법 3: 환경 변수로 모드 전환

```powershell
# API 모드로 전환
$env:LLM_MODE="api"
$env:GEMINI_API_KEY="your-api-key"

# 서버 실행
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 시작 로그 예시

### 로컬 서버 모드
```
[L.U.N.A. Startup] 서비스 초기화 시작...
[L.U.N.A. Startup] LLM 서비스 초기화 중...
[L.U.N.A. LLM Manager] 로컬 서버 모드로 초기화 완료
[L.U.N.A. Startup] LLM 모드: server, 타겟: rp
```

### Gemini API 모드
```
[L.U.N.A. Startup] 서비스 초기화 시작...
[L.U.N.A. Startup] LLM 서비스 초기화 중...
[L.U.N.A. LLM API] Gemini 클라이언트 초기화 완료
[L.U.N.A. LLM Manager] API 모드로 초기화 완료
[L.U.N.A. Startup] LLM 모드: api, 타겟: gemini
```

## 폴백 동작

LLM 설정 로드 실패 시:
```
[L.U.N.A. Startup] 경고: LLM 서비스 초기화 실패. 기본 서버 모드로 대체합니다.
```

기본 서버 설정으로 대체:
- `rp`: http://localhost:8080 (Luna)
- `translator`: http://localhost:8081 (Translator)

## 확인 방법

서버 시작 후 로그에서 다음을 확인하세요:
```
[L.U.N.A. Startup] LLM 모드: {mode}, 타겟: {target}
```

- `mode`: "server" 또는 "api"
- `target`: "rp" (서버 모드) 또는 "gemini" (API 모드)

## 트러블슈팅

### 1. Gemini API 키 오류
```
[L.U.N.A. LLM API] Gemini API 키가 설정되지 않았습니다.
```
해결: 환경 변수 `GEMINI_API_KEY` 설정

### 2. 설정 파일 로드 실패
```
[L.U.N.A. Config] 설정 파일을 찾을 수 없습니다: config/models.yaml
```
해결: `config/models.yaml` 파일 존재 확인

### 3. 서버 연결 실패 (서버 모드)
```
[L.U.N.A. LLM] 'rp' 서버 요청 중 오류 발생
```
해결: LLM 서버가 http://localhost:8080에서 실행 중인지 확인

## 추가 정보

상세한 사용 가이드는 `docs/LLM_MANAGER_GUIDE.md`를 참조하세요.
