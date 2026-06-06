"""chatbot_docs/*.md를 H2 섹션 단위로 청크화.

같은 Chroma 컬렉션에 type="doc_section" 메타로 합류.
파일명 → material 매핑:
  carbon_steel.md → carbon_steel
  stainless.md    → stainless
  aluminum.md     → aluminum
  posture_6G.md   → posture (position=6G)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from loader import Chunk, DATASET_DIR


DOCS_DIR = DATASET_DIR / "chatbot_docs"

FILE_TO_MATERIAL: dict[str, tuple[str, str]] = {
    "carbon_steel.md": ("carbon_steel", ""),
    "stainless.md":    ("stainless", ""),
    "aluminum.md":     ("aluminum", ""),
    "posture_6G.md":   ("posture", "6G"),
}

# 섹션 본문에서 자세 표기를 찾기 위한 패턴
POSITION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("1G", re.compile(r"\b1G\b")),
    ("2G", re.compile(r"\b2G\b")),
    ("5G", re.compile(r"\b5G\b")),
    ("6G", re.compile(r"\b6G\b")),
]

H2 = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def _split_h2(md: str) -> list[tuple[str, str]]:
    """(섹션 제목, 섹션 본문) 리스트. 최상단 H1 이전/직후는 버리고 ## 단위로만."""
    matches = list(H2.finditer(md))
    if not matches:
        return [("", md.strip())]
    sections: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        body = md[body_start:body_end].strip()
        sections.append((title, body))
    return sections


def _detect_position(text: str) -> str:
    found: list[str] = []
    for pos, pat in POSITION_PATTERNS:
        if pat.search(text):
            found.append(pos)
    # 단일 자세만 언급된 섹션이면 그걸 메타로. 여러 개면 빈 값(전체 비교 섹션).
    return found[0] if len(found) == 1 else ""


def load_doc_chunks(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.glob("*.md")):
        if path.name not in FILE_TO_MATERIAL:
            continue
        material, default_pos = FILE_TO_MATERIAL[path.name]
        md = path.read_text(encoding="utf-8")
        for sec_idx, (title, body) in enumerate(_split_h2(md)):
            if not body:
                continue
            position = default_pos or _detect_position(title + "\n" + body)
            stem = path.stem
            text = f"[{material}{'/' + position if position else ''}] {title}\n\n{body}".strip()
            chunks.append(Chunk(
                id=f"doc_{stem}_sec{sec_idx:02d}",
                text=text,
                material=material,
                position=position,
                type="doc_section",
                stage="",
                defect="",
                source_ids=f"summary_{material}" if material != "posture" else "summary_posture_6g",
                source_file=str(path.relative_to(DATASET_DIR)),
            ))
    return chunks


if __name__ == "__main__":
    chunks = load_doc_chunks()
    print(f"doc 청크 수: {len(chunks)}")
    for c in chunks:
        head = c.text.splitlines()[0]
        print(f"  {c.id:35s} {c.material:13s} {c.position:3s}  {head[:60]}")
