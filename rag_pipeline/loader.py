"""rag/*.json 파일을 읽어 임베딩용 청크 리스트로 변환."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"
RAG_DIR = DATASET_DIR / "rag"


@dataclass
class Chunk:
    id: str
    text: str
    material: str        # carbon_steel / stainless / aluminum / any
    position: str        # 1G / 2G / 5G / 6G
    type: str            # expert_tip / defect_solution / qa / parameters
    stage: str           # root_pass / fill_pass / ... or subtopic, "" if N/A
    defect: str          # defect name, "" if N/A
    source_ids: str      # comma-joined (Chroma metadata must be primitive)
    source_file: str     # relative path under dataset/

    def to_metadata(self) -> dict:
        meta = asdict(self)
        meta.pop("text")
        meta.pop("id")
        return meta


def _fmt_tip(material: str, position: str, stage: str, tip: str) -> str:
    head = f"[{material}/{position}]"
    if stage:
        head += f" {stage}"
    return f"{head}\n{tip}"


def _fmt_defect(material: str, position: str, defect: str, cause: str, solution: str) -> str:
    return (
        f"[{material}/{position}] 결함: {defect}\n"
        f"원인: {cause}\n"
        f"해결: {solution}"
    )


def _fmt_qa(material: str, position: str, q: str, a: str) -> str:
    return f"[{material}/{position}]\nQ: {q}\nA: {a}"


def _fmt_params(material: str, position: str, params: dict) -> str:
    lines = [f"[{material}/{position}] 표준 파라미터"]
    for k, v in params.items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)


def _load_one(path: Path) -> list[Chunk]:
    data = json.loads(path.read_text(encoding="utf-8"))
    material = data.get("material") or data.get("topic", "any")
    position = data.get("position", "")
    source_ids = ",".join(data.get("source_ids", []))
    source_file = str(path.relative_to(DATASET_DIR))
    stem = f"{material}_{position}" if position else material

    chunks: list[Chunk] = []

    if "parameters" in data and isinstance(data["parameters"], dict):
        chunks.append(Chunk(
            id=f"{stem}_params",
            text=_fmt_params(material, position, data["parameters"]),
            material=material,
            position=position,
            type="parameters",
            stage="",
            defect="",
            source_ids=source_ids,
            source_file=source_file,
        ))

    for idx, entry in enumerate(data.get("entries", [])):
        etype = entry.get("type", "")
        stage = entry.get("stage") or entry.get("subtopic", "")
        defect = entry.get("defect", "")

        if etype == "expert_tip":
            text = _fmt_tip(material, position, stage, entry.get("tip", ""))
        elif etype == "defect_solution":
            text = _fmt_defect(
                material, position,
                defect,
                entry.get("cause", ""),
                entry.get("solution", ""),
            )
        elif etype == "qa":
            text = _fmt_qa(
                material, position,
                entry.get("question", ""),
                entry.get("answer", ""),
            )
        else:
            # unknown type — embed raw JSON so it's still searchable
            text = json.dumps(entry, ensure_ascii=False)

        chunks.append(Chunk(
            id=f"{stem}_entry{idx:02d}",
            text=text,
            material=material,
            position=position,
            type=etype,
            stage=stage,
            defect=defect,
            source_ids=source_ids,
            source_file=source_file,
        ))

    return chunks


def load_all_chunks(rag_dir: Path = RAG_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(rag_dir.rglob("*.json")):
        chunks.extend(_load_one(path))
    return chunks


def iter_chunks(rag_dir: Path = RAG_DIR) -> Iterable[Chunk]:
    yield from load_all_chunks(rag_dir)


if __name__ == "__main__":
    chunks = load_all_chunks()
    print(f"총 청크 수: {len(chunks)}")
    by_material: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for c in chunks:
        by_material[c.material] = by_material.get(c.material, 0) + 1
        by_type[c.type] = by_type.get(c.type, 0) + 1
    print("재질별:", by_material)
    print("타입별:", by_type)
    print("\n예시 청크:")
    print(chunks[0].id)
    print(chunks[0].text)
    print("meta:", chunks[0].to_metadata())
