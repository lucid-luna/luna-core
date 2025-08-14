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
├── README.md
└── requirements.txt
</pre>

---

## API Reference

### 예시 API 리스트

#### Speech-to-Text (STT)

```http
  POST /api/stt/stream
```

| Parameter | Type   | Description                            |
| :-------- | :----- | :------------------------------------- |
| audio     | bytes  | **Required**. PCM 오디오 스트림         |
| language  | string | 언어 코드 (예: "ko-KR")                |
| api       | string | "azure" 또는 "whisper" 중 하나         |

#### Text Generation (LLM)

```http
  POST /api/llm/generate
```

| Parameter | Type   | Description                            |
| :-------- | :----- | :------------------------------------- |
| prompt    | string | **Required**. 프롬프트 텍스트           |
| model_id  | string | 사용할 모델 키 ("luna-main" 등)         |

#### Text-to-Speech (TTS)

```http
  POST /api/tts/speak
```

| Parameter | Type   | Description                            |
| :-------- | :----- | :------------------------------------- |
| text      | string | **Required**. 음성으로 변환할 텍스트    |
| voice     | string | 음성 프리셋 이름 ("luna", "mika" 등)    |

#### Vision Tagging

```http
  POST /api/vision/tag
```

| Parameter | Type   | Description                            |
| :-------- | :----- | :------------------------------------- |
| image     | file   | **Required**. 입력 이미지               |
| model_id  | string | 사용할 모델 ID ("lunavision")          |

#### Plugin: Web Search

```http
  POST /api/plugins/search
```

| Parameter | Type   | Description                            |
| :-------- | :----- | :------------------------------------- |
| query     | string | **Required**. 검색할 문장              |
| engine    | string | 검색 엔진 ("google", "duckduckgo" 등) |

---

## ⚙️ 향후 개발 계획

- [ ] STT 모듈 완성
- [ ] TTS 음성 출력 캐릭터 프리셋 및 감정 연동 구조 확장
- [ ] LLM WebSocket 스트리밍 처리 개선 및 캐릭터 메모리 연동
- [ ] Vision API → luna-models 연동
- [V] 캐릭터 상태 전송 시스템 구성
- [ ] Plugin 실행 구조 표준화 (search, calculate, spotify 등)
- [ ] /api/context 기반 세션 기억/복원 시스템 구축

---

## 🪪 라이선스

Apache License 2.0 © lucid-luna

---

## ✨ 프로젝트 전체 보기

👉 [lucid-luna/luna-core](https://github.com/lucid-luna/luna-core)  
👉 [lucid-luna/luna-client](https://github.com/lucid-luna/luna-client)  
👉 [lucid-luna/luna-plugins](https://github.com/lucid-luna/luna-plugins)
