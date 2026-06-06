"""Chroma persistent 컬렉션에 청크 + bge-m3 임베딩을 적재.

재실행 시 멱등하도록 컬렉션을 매번 재생성한다(청크가 100개 단위라 부담 없음).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import chromadb

from docs_loader import load_doc_chunks
from embedder import embed, MODEL_NAME
from loader import load_all_chunks


DEFAULT_DB_DIR = Path(__file__).resolve().parent / "chroma_db"
COLLECTION = "welding_rag"


def build(db_dir: Path = DEFAULT_DB_DIR) -> None:
    rag_chunks = load_all_chunks()
    doc_chunks = load_doc_chunks()
    chunks = rag_chunks + doc_chunks
    print(f"청크 {len(chunks)}개 로드 (rag={len(rag_chunks)}, docs={len(doc_chunks)})")

    db_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_dir))

    # 멱등: 있으면 지우고 다시 만든다
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION,
        metadata={"embedding_model": MODEL_NAME, "hnsw:space": "cosine"},
    )

    print("임베딩 생성 중...")
    vectors = embed([c.text for c in chunks])

    collection.add(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=vectors.tolist(),
        metadatas=[c.to_metadata() for c in chunks],
    )

    print(f"인덱스 저장 완료: {db_dir} (collection={COLLECTION}, count={collection.count()})")


def smoke_test(db_dir: Path = DEFAULT_DB_DIR) -> None:
    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_collection(COLLECTION)

    queries = [
        ("스테인 6G 백비드가 검게 나와요", {"material": "stainless", "position": "6G"}),
        ("탄소강 1G에서 와이어를 어디에 찍어야 하나요", {"material": "carbon_steel", "position": "1G"}),
        ("6G 자세에서 팔이 떨려요", {"position": "6G"}),
    ]
    for q, where in queries:
        q_vec = embed([q])[0].tolist()
        if len(where) > 1:
            chroma_where = {"$and": [{k: v} for k, v in where.items()]}
        else:
            chroma_where = where
        res = collection.query(
            query_embeddings=[q_vec],
            n_results=3,
            where=chroma_where,
        )
        print("\nQ:", q, " filter:", where)
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            print(f"  - dist={dist:.3f} [{meta.get('material')}/{meta.get('position')}/{meta.get('type')}]")
            first_line = doc.split("\n", 1)[0]
            print(f"    {first_line}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DEFAULT_DB_DIR, help="Chroma persistent dir")
    ap.add_argument("--test", action="store_true", help="빌드 후 샘플 쿼리 실행")
    args = ap.parse_args()

    build(args.db)
    if args.test:
        smoke_test(args.db)


if __name__ == "__main__":
    main()
