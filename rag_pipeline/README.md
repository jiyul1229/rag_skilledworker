# rag_pipeline

> 메인 README는 리포 루트의 [`../README.md`](../README.md). 여기는 이 폴더 안에서 보는 빠른 참조.

## 모듈 책임

| 파일 | 역할 |
|---|---|
| `loader.py` | `dataset/rag/*.json` → 청크 (메타: material/position/type/stage/defect) |
| `docs_loader.py` | `dataset/chatbot_docs/*.md` → H2 섹션 청크 |
| `embedder.py` | BAAI/bge-m3 임베딩 (sentence-transformers, GPU 자동 감지) |
| `build_index.py` | 청크 + 임베딩 → Chroma persistent 컬렉션 |
| `retriever.py` | 쿼리에서 material/position 룰 라우팅 + 메타 필터 벡터 검색 |
| `citations.py` | `sources.json` 인덱싱, source_id → 영상 메타 |
| **`rag_api.py`** | RAG의 단일 출구 — `(재질, 자세[, 쿼리]) → JSON` |
| **`agent.py`** | knowhow JSON + 관찰 → LLM → 구조화 피드백 |
| `chatbot.py` | 자유 텍스트 → 검색 + LLM → 자연어 답변 (데모용) |
| **`server.py`** | FastAPI — 위 기능 전부 HTTP 노출 |
| `cli.py` / `rag` | 통합 CLI 진입점 |
| `evaluate.py` | qa 청크 평가셋으로 Recall@k / MRR |

## 데이터 흐름

```
사용자 입력
    │
    ▼
업로드 + 가상분류  → (material, position)
    │
    ▼
rag_api.get_knowhow(material, position, query?)
    │   ↳ loader/docs_loader → Chroma → 메타 필터 + 의미 검색
    │   ↳ citations로 영상 인용 첨부
    │   ↳ 구조화 JSON 반환
    ▼
agent.generate_feedback(knowhow, observation)
    │   ↳ Groq or Gemini API (자동 감지)
    │   ↳ JSON 모드 응답 (summary/key_points/warnings/next_steps)
    ▼
사용자에게 표시 + 영상 인용
```

## LLM 백엔드 자동 감지

`chatbot._resolve_provider()`가 환경변수를 보고 결정:

1. `GROQ_API_KEY` 있으면 → Groq (`llama-3.3-70b-versatile` 기본)
2. `GEMINI_API_KEY` / `GOOGLE_API_KEY` / `GCP_PROJECT_ID`(AIzaSy로 시작) 있으면 → Gemini (`gemini-2.5-flash` 기본)
3. 둘 다 없으면 → dry-run (LLM 호출 스킵, 컨텍스트만 반환)

모델 강제 지정: `LLM_MODEL`, `GROQ_MODEL`, `VERTEX_MODEL` env.

## 자세한 CLI 예시는 [`USAGE.md`](USAGE.md)
