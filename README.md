<p align="center">
  <img src="https://github.com/lucid-luna/.github/blob/main/profile/assets/Butterfly.png" width="160" alt="LUNA Logo"/>
</p>

<h1 align="center">⚙️ Luna Core</h1>
<p align="center">
  <b>실시간 음성/시각 인터페이스와 멀티모달 AI 시스템을 위한 핵심 엔진</b><br/>
</p>

---

## 🧠 개요

`luna-core`는 [L.U.N.A.](https://github.com/lucid-luna) 프로젝트의 **핵심 기능**을 제공하는 백엔드 모듈입니다.

- 🎙️ **STT**: 실시간 음성 인식
- 🧠 **LLM**: 자연어 이해 및 응답 생성
- 🔊 **TTS**: 캐릭터 음성 합성 및 발화
- 👁️‍🗨️ **비전 입력 연동**: LunaModels 와 연동된 시각 추론 흐름
- 🔌 **플러그인 확장성**: 외부 기능 모듈 실행 (검색, 음악, 날씨 등)

---

## 📦 주요 구성 요소

<pre>
luna-core/
├── main.py
├── README.md
├── requirements.txt
├── config/
│   ├── models.yaml
│   └── models.yaml.sample
├── services/
│   ├── asr.py
│   ├── llm_api.py
│   ├── tts.py
│   ├── vision.py
│   ├── emotion.py
│   ├── multi_intent.py
│   ├── interaction.py
│   ├── cache.py
│   ├── translator.py
│   └── ...
├── models/
│   ├── emotion_model.py
│   ├── tts_model.py
│   ├── multiintent_model.py
│   └── ...
├── examples/
│   ├── quick_start.py
│   ├── llm_manager_example.py
│   └── ...
├── scripts/
│   └── migrate_memory.py
├── checkpoints/
├── cache/
├── memory/
├── outputs/
├── docs/
└── utils/
</pre>

---

## API Reference

### 실제 API 엔드포인트 요약

아래는 FastAPI 기반으로 구현된 주요 엔드포인트입니다. 실제 파라미터와 응답 구조는 코드(main.py 등)를 참고하세요.

#### 실시간 음성 인식 (ASR)

- **WebSocket** `/ws/asr` : PCM 오디오 스트림을 실시간으로 전송하면, 텍스트 변환 및 전체 파이프라인 결과를 반환합니다.

#### 텍스트 생성 (LLM)

- **POST** `/generate`
    - 요청: `{ input: str, temperature: float, max_tokens: int }`
    - 응답: `{ content: str }`

#### 텍스트 상호작용 (대화)

- **POST** `/interact`
    - 요청: `{ input: str }`
    - 응답: `{ ... }` (상호작용 결과)
- **POST** `/interact/stream`
    - 요청: `{ input: str }`
    - 응답: `text/event-stream` (실시간 스트리밍)

#### 음성 합성 (TTS)

- **POST** `/synthesize`
    - 요청: `{ text: str, style: str, style_weight: float }`
    - 응답: `{ audio_url: str }`
- **POST** `/synthesize/parallel`
    - 요청: `{ text: str, style: str, style_weight: float }`
    - 응답: `{ audio_url: str }` (병렬 처리)

#### 이미지 분석 (Vision)

- **POST** `/analyze/vision`
    - 요청: `file` (이미지 업로드)
    - 응답: `{ answer: str }`

#### 번역

- **POST** `/translate`
    - 요청: `{ text: str, from_lang: str, to_lang: str }`
    - 응답: `{ translated_text: str }`

#### 메모리 관리

- **GET** `/memory/stats` : 메모리 통계 조회
- **GET** `/memory/recent?count=10` : 최근 대화 내역 조회
- **DELETE** `/memory/clear` : 모든 대화 삭제
- **POST** `/memory/summarize` : 대화 요약 강제 실행
- **GET** `/memory/summary` : 현재 저장된 요약 반환

#### 캐시 관리

- **GET** `/cache/stats` : LLM 캐시 통계
- **POST** `/cache/cleanup` : 만료 캐시 정리
- **DELETE** `/cache/clear` : 모든 캐시 삭제

- **GET** `/tts/cache/stats` : TTS 캐시 통계
- **POST** `/tts/cache/cleanup` : 만료 TTS 캐시 정리
- **DELETE** `/tts/cache/clear` : 모든 TTS 캐시 삭제

#### 기타

- **GET** `/health` : 서버 상태 확인
- **GET** `/spotify/callback` : Spotify 인증 콜백

각 엔드포인트의 실제 파라미터/응답 구조는 main.py 및 각 서비스 파일을 참고하세요.

---

## ⚙️ 향후 개발 계획

- [x] STT 모듈 완성 (`services/asr.py`, `/ws/asr`)
- [x] TTS 음성 출력 캐릭터 프리셋 및 감정 연동 구조 (`services/tts.py`, `/synthesize`)
- [x] LLM WebSocket 스트리밍 처리 및 캐릭터 메모리 연동 (`services/llm_manager.py`, `/interact/stream`, `/memory/*`)
- [x] Vision API → luna-models 연동 (`services/vision.py`, `/analyze/vision`)
- [ ] 캐릭터 상태 전송 시스템 구성
- [x] Plugin 실행 구조 표준화 (search, calculate, spotify 등) (`services/mcp/tool_registry.py`, `/spotify/callback`)
- [ ] /api/context 기반 세션 기억/복원 시스템 구축

---

## 🪪 라이선스

Apache License 2.0 © lucid-luna

---

## ✨ 프로젝트 전체 보기

👉 [lucid-luna/luna-core](https://github.com/lucid-luna/luna-core)  
👉 [lucid-luna/luna-client](https://github.com/lucid-luna/luna-client)  
👉 [lucid-luna/luna-plugins](https://github.com/lucid-luna/luna-plugins)
