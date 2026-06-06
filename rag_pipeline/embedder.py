"""BAAI/bge-m3 임베딩 래퍼.

bge-m3는 한국어/영어 혼재 텍스트에 강하고, sentence-transformers로 쉽게 로드된다.
출력 차원: 1024.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np


MODEL_NAME = "BAAI/bge-m3"
EMBED_DIM = 1024


@lru_cache(maxsize=1)
def _get_model():
    # 지연 임포트: 인덱스 빌드/검색 시점에만 무거운 의존성 로드
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    return model


def embed(texts: Iterable[str], batch_size: int = 16, normalize: bool = True) -> np.ndarray:
    """텍스트 리스트를 (N, 1024) ndarray로 인코딩."""
    texts = list(texts)
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)
    model = _get_model()
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
        show_progress_bar=len(texts) > 32,
    )
    return vecs.astype(np.float32)


if __name__ == "__main__":
    sample = [
        "스테인 6G 백비드가 검게 나와요",
        "탄소강 1G에서 와이어를 어디에 찍어야 하나요",
    ]
    v = embed(sample)
    print("shape:", v.shape, "dtype:", v.dtype)
    # 코사인 유사도(정규화돼 있으므로 내적)
    print("sim:", float(v[0] @ v[1]))
