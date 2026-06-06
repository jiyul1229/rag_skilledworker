# Skilled Worker RAG — 파이프 TIG 용접 챗봇 백엔드

은퇴 숙련 용접공의 **노하우**를 신입에게 전달하기 위한 RAG 기반 챗봇/피드백 시스템.

> **2026 ICT 캡스톤** · 백엔드 + RAG 파트
> 분류된 (재질, 자세) → Vector DB 조회 → Agent가 신입에게 코칭 피드백 생성

---

## 1. 시스템 한눈에

```
프론트엔드 (브라우저)
    │  HTTP/JSON
    ▼
FastAPI 서버 (rag_pipeline/server.py, :8000)
    ├─ POST /api/upload        ─ 사진/영상 업로드 + 가상 분류
    ├─ POST /api/knowhow       ─ (재질, 자세) → 노하우 JSON
    ├─ POST /api/feedback      ─ Agent 피드백 (메인)
    ├─ GET  /api/training-videos
    ├─ GET  /api/video/{id}    ─ 영상 스트리밍
    └─ GET  /docs              ─ Swagger UI (자동 생성)
        │
        ├── rag_api.get_knowhow()      (재질·자세 → JSON)
        │     └── Chroma + bge-m3      (147 청크)
        ├── agent.generate_feedback()  (knowhow → LLM → 피드백)
        │     └── Groq (llama-3.3-70b)
        └── citations + sources.json   (영상 인용)
```

---

## 2. 빠른 시작 (5분)

### A. 클론 + 의존성

```bash
git clone https://github.com/jiyul1229/rag_skilledworker.git
cd rag_skilledworker

# Python 3.10+ 가상환경 (이미 venv 있으면 건너뛰기)
python3 -m venv ~/.venv
source ~/.venv/bin/activate
pip install -r rag_pipeline/requirements.txt
```

설치되는 주요 패키지: `chromadb`, `sentence-transformers`, `torch`, `google-genai`, `groq`, `fastapi`, `uvicorn`, `python-multipart`

### B. API 키 설정

```bash
cp rag_pipeline/.env.example rag_pipeline/.env
# .env 열어서 키 채우기
```

**`.env`에 둘 중 하나만 채우면 됨:**

```ini
# 옵션 1: Groq (추천 — 빠르고 무료 쿼터 큼)
GROQ_API_KEY=gsk_여기에본인키
# 키 발급: https://console.groq.com/keys

# 옵션 2: Gemini (Google AI Studio)
GEMINI_API_KEY=AIzaSy여기에본인키
# 키 발급: https://aistudio.google.com/app/apikey
```

코드가 자동으로 어떤 키가 있는지 감지해서 알아서 LLM 선택.

### C. 영상 파일 받기 (선택)

영상 14개(약 1.3GB)는 GitHub 용량 제한 때문에 별도 공유.
받아서 `dataset/video/` 에 넣으면 영상 재생 가능.
**없어도 RAG/Agent 기능은 정상 동작** (`/api/video/{id}`만 404).

자세한 내용: [`dataset/video/README.md`](dataset/video/README.md)

### D. 서버 띄우기

```bash
cd rag_pipeline
./rag serve
```

브라우저에서:
- **Swagger UI (모든 엔드포인트 시연):** http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- 헬스 체크: http://localhost:8000/api/health

> WSL2에서 localhost 안 잡히면 `hostname -I` 로 IP 확인 후 `http://172.x.x.x:8000/docs`

---

## 3. API 엔드포인트 (프론트엔드용)

| 메서드 | 경로 | 요청 | 응답 |
|---|---|---|---|
| `GET` | `/api/health` | - | `{"status":"ok"}` |
| `GET` | `/api/materials` | - | `["aluminum","carbon_steel","stainless"]` |
| `GET` | `/api/positions` | - | `["1G","2G","5G","6G"]` |
| `GET` | `/api/training-videos?material=&position=` | 쿼리스트링 | `[{id,title,material,position,video_url}]` |
| `GET` | `/api/sources` | - | 전체 영상 메타 |
| `GET` | `/api/video/{source_id}` | path param | mp4 스트리밍 (Range 지원) |
| `POST` | `/api/upload` | multipart `file` | `{upload_id, classification:{material,position,confidence,source}}` |
| `POST` | `/api/knowhow` | `{material,position,query?}` | parameters/tips/defects/qa/guides/citations 포함 JSON |
| `POST` | `/api/feedback` | `{material,position,observation?}` | **Agent 피드백** (메인) |
| `POST` | `/api/answer` | `{query}` | (데모) 자유 텍스트 Q&A |

### 프론트엔드 흐름 예시

```js
const API = "http://localhost:8000"

// 1) 사진 업로드 + 가상 분류
const fd = new FormData()
fd.append("file", fileInput.files[0])
const { classification } = await (await fetch(`${API}/api/upload`, {
  method: "POST", body: fd
})).json()
// classification = { material: "stainless", position: "6G", confidence, source }

// 2) Agent 피드백 받기
const fb = await (await fetch(`${API}/api/feedback`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    material: classification.material,
    position: classification.position,
    observation: "백비드가 검게 나와요",   // 사용자가 입력한 상황
  })
})).json()

// 화면에 렌더링:
// fb.feedback.summary       ← 진단 요약 (1~2문장)
// fb.feedback.key_points    ← 핵심 점검 [str]
// fb.feedback.warnings      ← 주의사항 [str]
// fb.feedback.next_steps    ← 다음 액션 [str]
// fb.citations              ← 근거 영상 [{id,title,video_url}]
// fb.training_videos        ← 같은 (재질,자세) 사전교육 영상 [{id,title,video_url}]
```

### 영상 재생

```jsx
<video controls src={`http://localhost:8000${citation.video_url}`} />
// citation.video_url = "/api/video/video_ss_6g" 같은 경로
```

### 응답 샘플 (실제 호출 결과)

```json
{
  "classification": { "material": "stainless", "position": "6G" },
  "observation": "백비드가 검게 나와요",
  "feedback": {
    "summary": "6시 부근에서 백퍼지 누설이나 가스 유량 부족이 원인입니다.",
    "key_points": [
      "백퍼지 가스 유량을 5~8 LPM로 조정",
      "마스킹 누설 점검",
      "6시 구간 전류 5~10A 하향",
      "진행 속도 균일 유지"
    ],
    "warnings": [
      "풀이 처지면 백퍼지가 빠져나갈 수 있음",
      "인터패스 온도 175°C 초과 시 결정립계 부식 위험"
    ],
    "next_steps": [
      "백퍼지 절차 재확인",
      "온도 크레용으로 인터패스 175°C 이하 확인",
      "6시 부근 와이어 한 번 더 찍어 풀 식히기"
    ]
  },
  "citations": [ { "id": "video_ss_6g", "title": "...", "video": "..." } ],
  "training_videos": [ ... ],
  "llm": { "provider": "groq", "model": "llama-3.3-70b-versatile", "dry_run": false }
}
```

---

## 4. 개발자용 CLI (서버 없이 빠른 테스트)

`rag_pipeline/` 안에서 `./rag <명령>` 한 줄.

```bash
./rag help                                   # 전체 도움말
./rag videos stainless 6G                    # 영상 카탈로그
./rag knowhow stainless 6G --query "백비드 검게"
./rag feedback stainless 6G "백비드가 검게 나와요"
./rag classify /path/to/사진.jpg              # 가상 분류
./rag search "스테인 6G 백비드 검게"          # 청크 검색만
./rag eval                                   # Recall@k / MRR
./rag serve                                  # 서버 띄우기
./rag build                                  # Chroma 인덱스 재빌드
```

기본 출력은 사람이 읽기 좋은 텍스트, `--json` 옵션 붙이면 raw JSON.

자세한 내용: [`rag_pipeline/USAGE.md`](rag_pipeline/USAGE.md)

---

## 5. 환경 변수 (`.env`)

| 변수명 | 필수 | 설명 |
|---|---|---|
| `GROQ_API_KEY` | △ | Groq API 키 (`gsk_...`). 있으면 우선 사용. |
| `GEMINI_API_KEY` | △ | Gemini API 키 (`AIzaSy...`). Groq 없을 때 fallback. |
| `GOOGLE_API_KEY` | △ | Gemini와 동일, 별칭. |
| `GCP_PROJECT_ID` | △ | `AIzaSy...` 형식이면 Gemini 키로 인식 (호환용). |
| `LLM_MODEL` | ✕ | 모델 강제 지정. 예: `llama-3.1-8b-instant` |
| `GROQ_MODEL` | ✕ | Groq 모델 (기본 `llama-3.3-70b-versatile`) |
| `VERTEX_MODEL` | ✕ | Gemini 모델 (기본 `gemini-2.5-flash`) |

**최소 설정**: `.env`에 `GROQ_API_KEY` 한 줄만 있으면 동작.
키가 없으면 자동으로 dry-run 모드 — feedback의 LLM 부분이 빈 배열로 떨어지고 나머지는 정상 동작.

---

## 6. 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| `ModuleNotFoundError: chromadb` | 시스템 파이썬 사용 | `~/.venv/bin/python` 또는 venv 활성화 |
| `chromadb.errors.CollectionNotFoundError` | 인덱스 없음 | `./rag build` |
| `address already in use` | 포트 8000 점유 | `./rag serve --port 8001` 또는 `pkill -f uvicorn` |
| feedback의 LLM이 빈 배열 | API 키 미설정 | `.env`에 `GROQ_API_KEY=` |
| `API key not valid` 에러 | 키 값이 형식 안 맞음 (예: 프로젝트 ID) | Groq는 `gsk_`, Gemini는 `AIzaSy`로 시작해야 |
| 브라우저 "failed to load page" | 서버 안 떠 있거나 WSL2 IP 문제 | `./rag serve --host 0.0.0.0` + WSL IP 직접 |
| `/api/video/{id}` 404 | 영상 파일 없음 | `dataset/video/`에 영상 파일 넣기 |

---

## 7. 디렉토리 구조

```
rag_skilledworker/
├── README.md                  ← 이 문서
├── .gitignore
├── dataset/                   ← 원본 데이터
│   ├── README.md              ← 데이터 설계 설명
│   ├── sources.json           ← 영상 ID ↔ 로컬 경로 매핑
│   ├── rag/                   ← 청크 단위 JSON (RAG 검색용)
│   │   ├── carbon_steel/{1G,2G,5G,6G}.json
│   │   ├── stainless/{1G,2G,6G}.json
│   │   ├── aluminum/{2G,5G}.json
│   │   └── posture/6G_posture.json
│   ├── chatbot_docs/          ← 마크다운 가이드 (LLM 컨텍스트용)
│   │   ├── carbon_steel.md
│   │   ├── stainless.md
│   │   ├── aluminum.md
│   │   └── posture_6G.md
│   └── video/                 ← 영상 mp4 (gitignored, 별도 공유)
└── rag_pipeline/              ← 백엔드 코드
    ├── README.md              ← 아키텍처 요약
    ├── USAGE.md               ← CLI 상세 사용법
    ├── requirements.txt
    ├── .env.example
    ├── rag                    ← bash 진입점 (chmod +x)
    ├── cli.py                 ← 통합 CLI
    ├── server.py              ← FastAPI 서버
    ├── rag_api.py             ← RAG 단일 출구 (재질,자세 → JSON)
    ├── agent.py               ← Agent 피드백 생성기
    ├── chatbot.py             ← 자유 텍스트 챗봇
    ├── retriever.py           ← 라우팅 + 벡터 검색
    ├── citations.py           ← 영상 인용 매핑
    ├── loader.py              ← rag/*.json → 청크
    ├── docs_loader.py         ← chatbot_docs/*.md → 청크
    ├── embedder.py            ← bge-m3 임베딩
    ├── build_index.py         ← Chroma 인덱스 빌드
    ├── evaluate.py            ← Recall@k / MRR
    └── chroma_db/             ← 사전 빌드된 벡터 인덱스 (2.2MB, 커밋됨)
```

---

## 8. 벤치마크 (현재)

평가셋: dataset의 qa 항목 28개

| 지표 | 값 |
|---|---|
| Recall@1 | **0.929** |
| Recall@3 | 0.929 |
| Recall@5 | 0.929 |
| MRR@5 | 0.929 |
| 라우팅 material acc | 0.393 |
| 라우팅 position acc | 0.679 |

재실행: `cd rag_pipeline && ./rag eval --show-misses`

---

## 9. 팀

| 파트 | 산출물 |
|---|---|
| 데이터셋 / 아키텍처 | `dataset/` (라벨링 JSON + 가이드 마크다운) |
| **RAG / 백엔드** | `rag_pipeline/` (이 리포의 본체) |
| 프론트엔드 | 별도 리포 — 이 README 참고해서 fetch |

문의는 이슈로.
