"""RAG → Agent 인터페이스 (JSON 출력).

전체 아키텍처에서 RAG 파트의 단일 출구. BE가 (material, position)을 분류해서 넘기면
관련 노하우를 구조화된 JSON으로 반환한다. Agent는 이 JSON을 받아 사용자용 report를 작성.

입력:
  material : carbon_steel | stainless | aluminum
  position : 1G | 2G | 5G | 6G
  query    : (선택) 자유 텍스트. 주면 의미 검색으로 청크 순위 재조정.
  top_k    : query가 있을 때 상위 몇 개를 더 강조할지.

출력 JSON 스키마:
{
  "material": str,
  "position": str,
  "query": str | null,
  "parameters": {...} | null,               # 표준 작업 파라미터
  "expert_tips": [{stage, tip, source_ids}],
  "defect_solutions": [{defect, cause, solution, source_ids}],
  "qa": [{question, answer, source_ids}],
  "guide_sections": [{title, body}],        # chatbot_docs의 H2 섹션
  "posture_notes": [...] | null,            # 6G면 자세 가이드 entries
  "citations": [{id, title, video, material, position}],
  "missing_videos": [str]
}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import citations
from loader import DATASET_DIR
from retriever import Hit, Retriever


VALID_MATERIALS = {"carbon_steel", "stainless", "aluminum"}
VALID_POSITIONS = {"1G", "2G", "5G", "6G"}


def _load_raw_json(material: str, position: str) -> dict | None:
    """원본 rag/<material>/<position>.json 그대로 로드 (없으면 None)."""
    path = DATASET_DIR / "rag" / material / f"{position}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_posture_raw() -> dict | None:
    path = DATASET_DIR / "rag" / "posture" / "6G_posture.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _entries_by_type(entries: list[dict], etype: str) -> list[dict]:
    return [e for e in entries if e.get("type") == etype]


def _rerank_indices(query: str, items: list[str], retriever: Retriever, k: int) -> list[int]:
    """items(텍스트 리스트)를 query와의 유사도 순으로 상위 k개 인덱스 반환."""
    from embedder import embed
    if not items:
        return []
    vecs = embed([query] + items)
    q = vecs[0]
    sims = [(i, float(vecs[i + 1] @ q)) for i in range(len(items))]
    sims.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in sims[:k]]


def _doc_sections(material: str, retriever: Retriever, query: str | None, k: int) -> list[dict]:
    """chatbot_docs/<material>.md의 H2 섹션 청크들. query 있으면 의미 검색, 없으면 메타 필터로 전체."""
    from embedder import embed

    where: dict
    where = {"$and": [{"type": "doc_section"}, {"material": material}]}
    if query:
        q_vec = embed([query])[0].tolist()
        res = retriever.collection.query(query_embeddings=[q_vec], n_results=k, where=where)
    else:
        # 메타 필터만 — 전부 가져옴
        res = retriever.collection.get(where=where)
        # get() 결과 포맷 통일
        res = {"ids": [res["ids"]], "documents": [res["documents"]], "metadatas": [res["metadatas"]]}

    out: list[dict] = []
    if not res.get("documents") or not res["documents"][0]:
        return out
    for doc in res["documents"][0]:
        lines = doc.splitlines()
        # 첫 줄은 "[material/...] 제목" 형태
        title = lines[0] if lines else ""
        body = "\n".join(lines[2:]).strip() if len(lines) > 2 else ""
        out.append({"title": title.strip(), "body": body})
    return out


def get_knowhow(
    material: str,
    position: str,
    query: str | None = None,
    top_k: int = 5,
    include_posture: bool | None = None,
) -> dict[str, Any]:
    if material not in VALID_MATERIALS:
        raise ValueError(f"material must be one of {sorted(VALID_MATERIALS)}, got {material!r}")
    if position not in VALID_POSITIONS:
        raise ValueError(f"position must be one of {sorted(VALID_POSITIONS)}, got {position!r}")

    raw = _load_raw_json(material, position) or {}
    entries: list[dict] = raw.get("entries", [])
    parameters = raw.get("parameters")
    source_ids_combined: list[str] = list(raw.get("source_ids", []))

    expert_tips = [
        {"stage": e.get("stage", ""), "tip": e.get("tip", "")}
        for e in _entries_by_type(entries, "expert_tip")
    ]
    defect_solutions = [
        {
            "defect": e.get("defect", ""),
            "cause": e.get("cause", ""),
            "solution": e.get("solution", ""),
        }
        for e in _entries_by_type(entries, "defect_solution")
    ]
    qa = [
        {"question": e.get("question", ""), "answer": e.get("answer", "")}
        for e in _entries_by_type(entries, "qa")
    ]

    retriever = Retriever()

    # query가 있으면 각 섹션에서 상위 top_k만 남김
    if query:
        if expert_tips:
            idxs = _rerank_indices(query, [t["tip"] for t in expert_tips], retriever, top_k)
            expert_tips = [expert_tips[i] for i in idxs]
        if defect_solutions:
            idxs = _rerank_indices(
                query,
                [f"{d['defect']}: {d['cause']} → {d['solution']}" for d in defect_solutions],
                retriever, top_k,
            )
            defect_solutions = [defect_solutions[i] for i in idxs]
        if qa:
            idxs = _rerank_indices(query, [q["question"] for q in qa], retriever, top_k)
            qa = [qa[i] for i in idxs]

    guide_sections = _doc_sections(material, retriever, query, k=top_k)

    # 6G면 posture 가이드 entries도 포함
    if include_posture is None:
        include_posture = (position == "6G")
    posture_notes: list[dict] | None = None
    if include_posture:
        praw = _load_posture_raw() or {}
        p_entries = praw.get("entries", [])
        posture_notes = [
            {
                "subtopic": e.get("subtopic", ""),
                "type": e.get("type", ""),
                "tip": e.get("tip", "") or e.get("answer", ""),
                "question": e.get("question", ""),
                "defect": e.get("defect", ""),
                "cause": e.get("cause", ""),
                "solution": e.get("solution", ""),
            }
            for e in p_entries
        ]
        source_ids_combined += list(praw.get("source_ids", []))

    cits = citations.resolve(source_ids_combined)

    return {
        "material": material,
        "position": position,
        "query": query,
        "parameters": parameters,
        "expert_tips": expert_tips,
        "defect_solutions": defect_solutions,
        "qa": qa,
        "guide_sections": guide_sections,
        "posture_notes": posture_notes,
        "citations": [
            {"id": c.id, "title": c.title, "video": c.video,
             "material": c.material, "position": c.position}
            for c in cits
        ],
        "missing_videos": citations.missing_videos(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", required=True, choices=sorted(VALID_MATERIALS))
    ap.add_argument("--position", required=True, choices=sorted(VALID_POSITIONS))
    ap.add_argument("--query", default=None, help="(선택) 자유 텍스트 쿼리로 청크 순위 재조정")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--no-posture", action="store_true", help="6G여도 posture 청크 제외")
    ap.add_argument("--out", type=Path, default=None, help="JSON 저장 경로. 없으면 stdout")
    args = ap.parse_args()

    payload = get_knowhow(
        material=args.material,
        position=args.position,
        query=args.query,
        top_k=args.top_k,
        include_posture=(False if args.no_posture else None),
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"saved → {args.out}  ({len(text)} chars)")
    else:
        print(text)


if __name__ == "__main__":
    main()
