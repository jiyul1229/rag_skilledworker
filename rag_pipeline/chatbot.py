"""사용자 질문 → 검색 → 컨텍스트 조립 → Gemini 응답 + 인용.

흐름:
  1. retriever.search(k)으로 짧은 청크 top-k 회수 (rag/*.json + chatbot_docs 통합)
  2. 라우팅된 material에 맞춰 해당 가이드 문서 청크도 추가
  3. 시스템 프롬프트(역할/규칙) + 사용자 메시지(컨텍스트+질문)
  4. citations.format_markdown으로 영상 인용 첨부

API 키 탐색 순서 (env):
  GCP_PROJECT_ID → GEMINI_API_KEY → GOOGLE_API_KEY
모델: VERTEX_MODEL env (기본 gemini-2.5-flash)

키가 없으면 어셈블된 프롬프트만 출력하는 dry-run 모드.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import citations
from retriever import Hit, Retriever, RouteDecision


def _load_dotenv() -> None:
    """rag_pipeline/.env를 읽어 os.environ에 주입 (이미 있는 키는 건드리지 않음)."""
    env_file = Path(__file__).resolve().parent / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and not os.environ.get(k):
            os.environ[k] = v


_load_dotenv()


MAX_RAG_HITS = 5
MAX_DOC_HITS = 3

# 키 자동 감지 우선순위
GROQ_ENV = "GROQ_API_KEY"
GEMINI_ENVS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GCP_PROJECT_ID")

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

API_KEY_ENVS = (GROQ_ENV,) + GEMINI_ENVS  # 외부 노출용 (CLI 메시지)


def _looks_like_gemini_key(v: str) -> bool:
    """진짜 Gemini API 키만 통과시킴 (프로젝트 ID 같은 거 걸러냄)."""
    return v.startswith("AIzaSy") and len(v) >= 30


def _resolve_provider() -> tuple[str | None, str, str | None, str]:
    """(api_key, provider, env_name, model)를 반환.
    provider ∈ {"groq", "gemini", "none"}.
    """
    model_override = os.environ.get("LLM_MODEL")

    groq_key = os.environ.get(GROQ_ENV)
    if groq_key:
        model = model_override or os.environ.get("GROQ_MODEL") or DEFAULT_GROQ_MODEL
        return groq_key, "groq", GROQ_ENV, model

    for name in GEMINI_ENVS:
        v = os.environ.get(name)
        if v and _looks_like_gemini_key(v):
            model = model_override or os.environ.get("VERTEX_MODEL") or DEFAULT_GEMINI_MODEL
            return v, "gemini", name, model

    # 기본 모델 (메시지용)
    return None, "none", None, model_override or DEFAULT_GROQ_MODEL


# 하위 호환: 기존 함수 시그니처 유지
def _get_api_key() -> tuple[str | None, str | None]:
    key, _, name, _ = _resolve_provider()
    return key, name


def _get_model() -> str:
    _, _, _, model = _resolve_provider()
    return model


SYSTEM_PROMPT = """당신은 파이프 TIG 용접 챗봇이다. 은퇴한 숙련 용접공의 관점에서 신입에게 가르치듯 답한다.

규칙:
- 반드시 제공된 <context> 안의 정보만 사용해 답한다. 추측·일반론·교과서 지식 금지.
- 컨텍스트에 답이 없으면 "제공된 자료에서 확인할 수 없습니다"라고 말한다.
- 재질(탄소강/스테인리스/알루미늄)과 자세(1G/2G/5G/6G)를 명확히 구분해서 답한다.
- 숫자(전류, 가스 LPM, 갭, 각도)는 컨텍스트의 값을 그대로 인용한다.
- 답은 한국어로, 3~6문장 또는 핵심 bullet 4~6개 이내로 간결하게.
- 마지막에 어떤 청크 id를 근거로 했는지 `근거: [id1, id2]` 형태로 한 줄 표기.
"""


@dataclass
class Answer:
    text: str
    hits: list[Hit]
    decision: RouteDecision
    citations_md: str


def _format_context(hits: list[Hit]) -> str:
    blocks: list[str] = []
    for i, h in enumerate(hits, 1):
        head = f"[{i}] id={h.id} | {h.material}/{h.position}/{h.type}"
        if h.stage:
            head += f" · {h.stage}"
        if h.defect:
            head += f" · defect={h.defect}"
        blocks.append(f"{head}\n{h.text}")
    return "\n\n---\n\n".join(blocks)


def _doc_hits(retriever: Retriever, query: str, decision: RouteDecision, k: int) -> list[Hit]:
    """라우팅된 material에 맞는 doc_section 청크 보강 검색.

    doc 청크 대부분이 재질 전체 가이드(position="")라 자세 필터는 걸지 않는다.
    posture 도메인 질문이면 자동으로 material=posture로 라우팅돼 6G 문서를 가져옴.
    """
    where_clauses: list[dict] = [{"type": "doc_section"}]
    if decision.material:
        where_clauses.append({"material": decision.material})
    where: dict = where_clauses[0] if len(where_clauses) == 1 else {"$and": where_clauses}

    from embedder import embed
    q_vec = embed([query])[0].tolist()
    res = retriever.collection.query(query_embeddings=[q_vec], n_results=k, where=where)
    from retriever import _to_hits
    return _to_hits(res)


def answer(query: str, k: int = MAX_RAG_HITS, dry_run: bool = False) -> Answer:
    r = Retriever()
    hits, decision = r.search(query, k=k)

    # 가이드 문서 보강 (doc_section만 따로 메타 필터로)
    doc_hits = _doc_hits(r, query, decision, k=MAX_DOC_HITS)

    # rag 청크 우선, doc 청크는 뒤에. id 중복 제거.
    seen: set[str] = set()
    merged: list[Hit] = []
    for h in hits + doc_hits:
        if h.id in seen:
            continue
        seen.add(h.id)
        merged.append(h)

    context = _format_context(merged)
    user_msg = (
        f"<context>\n{context}\n</context>\n\n"
        f"질문: {query}"
    )

    cits = citations.for_hits(merged)
    cits_md = citations.format_markdown(cits)

    api_key, provider, env_used, model = _resolve_provider()
    if dry_run or not api_key:
        reason = "DRY-RUN" if dry_run else "API 키 미설정 (env: GROQ_API_KEY 또는 AIzaSy...로 시작하는 GEMINI/GOOGLE/GCP_PROJECT_ID)"
        text = (
            f"[{reason}]\n\n"
            f"=== MODEL: {model} (provider={provider}) ===\n"
            f"=== SYSTEM ===\n{SYSTEM_PROMPT}\n"
            f"=== USER ===\n{user_msg}\n"
        )
        return Answer(text=text, hits=merged, decision=decision, citations_md=cits_md)

    if provider == "groq":
        text = _call_groq(SYSTEM_PROMPT, user_msg, api_key, model)
    else:
        text = _call_gemini(SYSTEM_PROMPT, user_msg, api_key, model)
    return Answer(text=text, hits=merged, decision=decision, citations_md=cits_md)


def _call_groq(system: str, user: str, api_key: str, model: str) -> str:
    try:
        from groq import Groq
    except ImportError as e:
        return f"[groq SDK 미설치 — `pip install groq`] ({e})"

    client = Groq(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=1024,
        )
    except Exception as e:
        return f"[Groq 호출 실패 — model={model}] {type(e).__name__}: {e}"
    return (resp.choices[0].message.content or "").strip() or "[Groq가 빈 응답을 반환했습니다]"


def _call_gemini(system: str, user: str, api_key: str, model: str) -> str:
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        return f"[google-genai SDK 미설치 — `pip install google-genai`] ({e})"

    client = genai.Client(api_key=api_key)
    try:
        resp = client.models.generate_content(
            model=model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=1024,
                temperature=0.2,
            ),
        )
    except Exception as e:
        return f"[Gemini 호출 실패 — model={model}] {type(e).__name__}: {e}"

    return (resp.text or "").strip() or "[Gemini가 빈 응답을 반환했습니다]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="용접 관련 자연어 질문")
    ap.add_argument("-k", type=int, default=MAX_RAG_HITS)
    ap.add_argument("--dry-run", action="store_true", help="LLM 호출 없이 프롬프트만 출력")
    args = ap.parse_args()

    a = answer(args.query, k=args.k, dry_run=args.dry_run)

    print(f"# 질문\n{args.query}\n")
    print(f"# 라우팅: {a.decision.reason}\n")
    print(f"# 답변\n{a.text}\n")
    print("# 참고 영상")
    print(a.citations_md)


if __name__ == "__main__":
    main()
