"""쿼리 → (material, position) 라우팅 + Chroma 메타 필터 검색.

룰 기반 라우팅:
- 재질 키워드 사전 (한/영/약어)
- 자세 키워드 사전 (1G~6G, "아래보기/수평/수직/45도/고정관" 등)
- 자세(posture) 도메인 키워드(팔/어깨/호흡/시선 등)가 강하면 material="posture" 로 강제

라우팅 실패 시 필터 없이 전체 검색 fallback.
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import chromadb

from embedder import embed


DEFAULT_DB_DIR = Path(__file__).resolve().parent / "chroma_db"
COLLECTION = "welding_rag"


MATERIAL_KEYWORDS: dict[str, list[str]] = {
    "carbon_steel": ["탄소강", "탄소 강", "carbon steel", "carbon_steel", "cs강", " cs ", "연강"],
    "stainless":    ["스테인", "스텐", "스뎅", "sus", "stainless", "ss강", "스테인레스", "스테인리스"],
    "aluminum":     ["알루미늄", "알미늄", "알류미늄", "aluminum", "aluminium", " al "],
}

POSITION_KEYWORDS: dict[str, list[str]] = {
    "1G": ["1g", "아래보기", "회전관", "파이프 회전"],
    "2G": ["2g", "수평", "horizontal"],
    "5G": ["5g", "수직 상진", "수직상진", "vertical up", "고정관 수직"],
    "6G": ["6g", "45도", "45°", "사십오도", "경사 고정관", "고정관 45"],
}

# 이 키워드들이 등장하면 자세 도메인으로 강제 라우팅
POSTURE_KEYWORDS: list[str] = [
    "자세", "팔꿈치", "팔이", "어깨", "호흡", "시선", "무게중심",
    "무릎", "발", "다리", "허리", "마스크", "떨", "흔들",
    "낮은자세", "높은자세", "기본자세", "모재 높이", "높이 설정",
]


@dataclass
class RouteDecision:
    material: str | None
    position: str | None
    is_posture: bool
    reason: str


# 재질명과 글자가 겹쳐 오탐을 일으키는 일반 용접 용어들 → 매칭 전 제거
# (예: "텅스텐"의 "스텐"이 stainless 키워드 "스텐"과 충돌)
_ROUTE_NOISE = ["텅스텐", "tungsten"]


def route_query(q: str) -> RouteDecision:
    qn = " " + q.lower() + " "
    for noise in _ROUTE_NOISE:
        qn = qn.replace(noise, " ")
    reasons: list[str] = []

    material: str | None = None
    for mat, kws in MATERIAL_KEYWORDS.items():
        if any(kw.lower() in qn for kw in kws):
            material = mat
            reasons.append(f"material={mat}")
            break

    position: str | None = None
    for pos, kws in POSITION_KEYWORDS.items():
        if any(kw.lower() in qn for kw in kws):
            position = pos
            reasons.append(f"position={pos}")
            break

    is_posture = any(kw in q for kw in POSTURE_KEYWORDS)

    # 자세 키워드가 있고 6G면 posture 컬렉션으로 강제
    # (재질이 명시되지 않았거나 자세 신호가 강하면)
    if is_posture and (material is None or position == "6G"):
        material = "posture"
        reasons.append("forced→posture")

    return RouteDecision(
        material=material,
        position=position,
        is_posture=is_posture,
        reason=", ".join(reasons) or "no-match",
    )


def _build_where(material: str | None, position: str | None) -> dict | None:
    clauses: list[dict] = []
    if material:
        clauses.append({"material": material})
    if position:
        clauses.append({"position": position})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


@dataclass
class Hit:
    id: str
    text: str
    score: float        # 1 - cosine_distance (높을수록 유사)
    material: str
    position: str
    type: str
    stage: str
    defect: str
    source_ids: list[str]
    source_file: str


class Retriever:
    def __init__(self, db_dir: Path = DEFAULT_DB_DIR, collection: str = COLLECTION):
        self.client = chromadb.PersistentClient(path=str(db_dir))
        self.collection = self.client.get_collection(collection)

    def search(
        self,
        query: str,
        k: int = 5,
        material: str | None = None,
        position: str | None = None,
        auto_route: bool = True,
    ) -> tuple[list[Hit], RouteDecision]:
        if auto_route and material is None and position is None:
            decision = route_query(query)
            material, position = decision.material, decision.position
        else:
            decision = RouteDecision(material, position, False, "manual")

        q_vec = embed([query])[0].tolist()
        where = _build_where(material, position)

        res = self.collection.query(
            query_embeddings=[q_vec],
            n_results=k,
            where=where,
        )
        hits = _to_hits(res)

        # 필터 결과가 비면 필터 없이 재시도
        if not hits and where is not None:
            res = self.collection.query(query_embeddings=[q_vec], n_results=k)
            hits = _to_hits(res)
            decision = RouteDecision(material, position, decision.is_posture,
                                     decision.reason + " → fallback no-filter")

        return hits, decision


def _to_hits(res: dict) -> list[Hit]:
    if not res["ids"] or not res["ids"][0]:
        return []
    hits: list[Hit] = []
    for id_, doc, meta, dist in zip(
        res["ids"][0],
        res["documents"][0],
        res["metadatas"][0],
        res["distances"][0],
    ):
        hits.append(Hit(
            id=id_,
            text=doc,
            score=round(1.0 - float(dist), 4),
            material=meta.get("material", ""),
            position=meta.get("position", ""),
            type=meta.get("type", ""),
            stage=meta.get("stage", ""),
            defect=meta.get("defect", ""),
            source_ids=[s for s in meta.get("source_ids", "").split(",") if s],
            source_file=meta.get("source_file", ""),
        ))
    return hits


def _print_hit(h: Hit) -> None:
    head = f"  [{h.score:.3f}] {h.material}/{h.position}/{h.type}"
    if h.stage:
        head += f" · {h.stage}"
    if h.defect:
        head += f" · defect={h.defect}"
    print(head)
    print(f"    src: {h.source_file}  ids={h.source_ids}")
    for line in h.text.splitlines():
        print(f"    {line}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", help="검색할 자연어 질문")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--material")
    ap.add_argument("--position")
    ap.add_argument("--demo", action="store_true", help="샘플 쿼리 5개 실행")
    args = ap.parse_args()

    r = Retriever()

    if args.demo or not args.query:
        samples = [
            "스테인 6G 백비드가 검게 나와요",
            "탄소강 1G에서 와이어를 어디에 찍어야 하나요",
            "알루미늄 2G에서 텅스텐이 자꾸 오염돼요",
            "6G에서 팔이 떨려요",
            "고정관 45도 자세에서 호흡은 어떻게 하나요",
        ]
        for q in samples:
            hits, dec = r.search(q, k=3)
            print(f"\nQ: {q}")
            print(f"  route: {dec.reason}")
            for h in hits:
                _print_hit(h)
        return

    hits, dec = r.search(args.query, k=args.k, material=args.material, position=args.position)
    print(f"Q: {args.query}")
    print(f"route: {dec.reason}\n")
    for h in hits:
        _print_hit(h)


if __name__ == "__main__":
    main()
