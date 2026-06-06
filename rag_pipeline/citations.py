"""sources.json 기반 인용(citation) 조립.

각 청크의 source_ids → sources.json의 영상 엔트리로 매핑한다.
missing_videos에 있는 항목은 인용 대상에서 제외.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from loader import DATASET_DIR
from retriever import Hit


SOURCES_JSON = DATASET_DIR / "sources.json"


@dataclass
class Citation:
    id: str
    title: str
    video: str         # dataset/ 기준 상대 경로
    material: str
    position: str


@lru_cache(maxsize=1)
def _load_sources() -> tuple[dict[str, Citation], list[str]]:
    data = json.loads(SOURCES_JSON.read_text(encoding="utf-8"))
    by_id: dict[str, Citation] = {}
    for s in data.get("sources", []):
        if s.get("kind") != "video":
            continue
        by_id[s["id"]] = Citation(
            id=s["id"],
            title=s.get("title", ""),
            video=s.get("video", ""),
            material=s.get("material", ""),
            position=s.get("position", ""),
        )
    missing = list(data.get("missing_videos", []))
    return by_id, missing


def resolve(source_ids: list[str]) -> list[Citation]:
    """source_ids 리스트를 Citation 리스트로. 매핑 없는 id(요약/미수집)는 스킵."""
    by_id, _ = _load_sources()
    cits: list[Citation] = []
    seen: set[str] = set()
    for sid in source_ids:
        if sid in seen:
            continue
        seen.add(sid)
        if sid in by_id:
            cits.append(by_id[sid])
    return cits


def for_hits(hits: list[Hit]) -> list[Citation]:
    """Hit 리스트 전체에서 등장한 인용을 중복 제거해 반환."""
    seen: set[str] = set()
    out: list[Citation] = []
    for h in hits:
        for c in resolve(h.source_ids):
            if c.id in seen:
                continue
            seen.add(c.id)
            out.append(c)
    return out


def format_markdown(cits: list[Citation]) -> str:
    if not cits:
        return "_(인용 가능한 로컬 영상 없음)_"
    lines = []
    for i, c in enumerate(cits, 1):
        tag = f"[{c.material or '-'}/{c.position or '-'}]"
        lines.append(f"{i}. **{c.title}** {tag}\n   `{c.video}`")
    return "\n".join(lines)


def missing_videos() -> list[str]:
    _, missing = _load_sources()
    return missing


if __name__ == "__main__":
    by_id, missing = _load_sources()
    print(f"인덱싱된 영상: {len(by_id)}")
    for c in by_id.values():
        print(f"  {c.id:25s} {c.material:13s} {c.position:4s} {c.title[:50]}")
    print(f"\nmissing_videos ({len(missing)}):")
    for m in missing:
        print(f"  - {m}")

    # 샘플: 가짜 Hit으로 인용 매핑 확인
    print("\n샘플 resolve:")
    sample_ids = ["video_ss_6g", "summary_stainless", "summary_posture_6g", "video_posture_high"]
    for c in resolve(sample_ids):
        print(f"  {c.id} → {c.video}")
