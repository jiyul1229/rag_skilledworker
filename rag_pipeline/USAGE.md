# RAG 파이프라인 사용법

> 모든 기능은 **`./rag <명령어>`** 한 줄로 호출.
> 처음 한 번만:
> ```bash
> cd ~/2026_ICT/rag_pipeline
> ```
> 그 이후엔 그 디렉토리에서 `./rag ...` 으로 다.

---

## 한눈에 보는 명령어

| 명령 | 하는 일 |
|---|---|
| `./rag help` | 전체 도움말 |
| `./rag videos [재질] [자세]` | 영상 카탈로그 |
| `./rag knowhow <재질> <자세>` | 노하우 보기 (예쁜 출력) |
| `./rag knowhow <재질> <자세> --query "..."` | 노하우 + 쿼리 순위 재조정 |
| `./rag feedback <재질> <자세> "상황 설명"` | Agent 피드백 |
| `./rag search "자유 텍스트"` | 청크 검색 |
| `./rag classify <파일경로>` | 가상 분류 |
| `./rag chat "자유 텍스트"` | Gemini 챗봇 답변 |
| `./rag eval` | 검색 품질 평가 (Recall/MRR) |
| `./rag serve` | FastAPI 서버 띄우기 (프론트엔드용) |
| `./rag build` | Chroma 인덱스 빌드 |

> 어떤 명령이든 끝에 `--json` 붙이면 raw JSON, 안 붙이면 사람용 텍스트.

---

## 자주 쓰는 시연 5종

### 1. 영상 목록 (사전교육)
```bash
./rag videos                       # 전체 14개
./rag videos stainless 6G          # 필터
```

### 2. 노하우 (RAG 핵심 출구)
```bash
./rag knowhow stainless 6G
./rag knowhow stainless 6G --query "백비드 검게"
./rag knowhow aluminum 2G --query "텅스텐 오염"
```

### 3. Agent 피드백 (실시간 피드백 파이프라인)
```bash
./rag feedback stainless 6G "백비드가 검게 나와요"
./rag feedback carbon_steel 5G "5G 6시 구간에서 풀이 떨어져요"

# API 키 없을 땐 --dry-run으로 컨텍스트만 확인
./rag feedback stainless 6G "백비드 검게" --dry-run
```

### 4. 가상 분류 (업로드 시뮬)
```bash
./rag classify /path/to/사진.jpg              # 파일명 힌트로 추정
./rag classify foo.jpg --material stainless --position 6G   # 강제 지정
```

### 5. 검색만 (디버깅용)
```bash
./rag search "스테인 6G 백비드 검게"
./rag search "캡 패스 언더컷" --material carbon_steel --position 5G
```

---

## Gemini API 키 (피드백/챗봇용)

```bash
export GCP_PROJECT_ID="여기에API키"
# 또는: export GEMINI_API_KEY=... / export GOOGLE_API_KEY=...
export VERTEX_MODEL="gemini-2.5-flash"   # 선택 (기본값)
```

키 안 넣으면 `feedback`/`chat`은 자동으로 dry-run으로 떨어져 (빈 답변 + 컨텍스트만).

---

## 프론트엔드에 넘기는 법

```bash
./rag serve
# → http://localhost:8000/docs  (Swagger UI, 프론트팀에게 이 링크만)
```

서버 띄우면 같은 기능이 HTTP로:

| CLI | HTTP |
|---|---|
| `./rag videos m p` | `GET /api/training-videos?material=m&position=p` |
| `./rag knowhow m p` | `POST /api/knowhow` |
| `./rag feedback m p "..."` | `POST /api/feedback` |
| `./rag classify file` | `POST /api/upload` (multipart) |

---

## 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| `./rag: Permission denied` | 실행 권한 | `chmod +x rag` |
| `./rag: command not found` | 디렉토리 안 들어옴 | `cd ~/2026_ICT/rag_pipeline` |
| `ModuleNotFoundError: chromadb` | venv 우회됨 | rag 스크립트가 자동으로 `~/.venv` 사용함. 그래도 뜨면 `~/.venv/bin/pip install -r requirements.txt` |
| `chromadb.errors.CollectionNotFoundError` | 인덱스 없음 | `./rag build` |
| feedback 응답이 비어있음 | API 키 없음 | `export GCP_PROJECT_ID=...` |
| `address already in use` | 포트 8000 점유 | `./rag serve --port 8001` |

---

## 파일 한눈에

```
rag_pipeline/
├── rag                ← ★ 통합 CLI 진입점 (bash 스크립트)
├── cli.py             ← CLI 본체 (argparse 서브커맨드)
├── server.py          ← FastAPI 서버 (프론트용)
├── rag_api.py         ← (재질, 자세) → 노하우 JSON
├── agent.py           ← 노하우 → Agent 피드백
├── chatbot.py         ← 자유 Q&A 데모
├── retriever.py       ← 라우팅 + 검색
├── citations.py       ← 영상 인용
├── loader.py          ← rag/*.json 청크
├── docs_loader.py     ← chatbot_docs/*.md 청크
├── embedder.py        ← bge-m3
├── build_index.py     ← Chroma 인덱스 빌드
├── evaluate.py        ← Recall/MRR
├── chroma_db/         ← 벡터 인덱스
├── uploads/           ← 업로드 파일 저장
└── requirements.txt
```
