"""qa 청크를 평가셋으로 활용해 RAG 검색 품질 측정.

각 qa 청크의 질문을 retriever에 던지고, top-k 결과에 자기 자신(id) 또는
같은 (material, position)의 다른 청크가 들어오는지 측정.

지표:
- Recall@1, @3, @5: 정답 청크 id가 top-k에 들어온 비율
- MRR: 1/rank 평균 (top-k 안에 없으면 0)
- 라우팅 정확도: 라우팅된 material/position이 정답과 일치하는 비율
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass

from loader import load_all_chunks
from retriever import Retriever, route_query


@dataclass
class Sample:
    qid: str
    question: str
    material: str
    position: str


def _extract_question(text: str) -> str:
    # qa 청크 텍스트 포맷: "[mat/pos]\nQ: ...\nA: ..."
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("Q:"):
            return s[2:].strip()
    return text.splitlines()[-1]


def build_eval_set() -> list[Sample]:
    samples: list[Sample] = []
    for c in load_all_chunks():
        if c.type != "qa":
            continue
        samples.append(Sample(
            qid=c.id,
            question=_extract_question(c.text),
            material=c.material,
            position=c.position,
        ))
    return samples


def evaluate(k: int = 5, verbose: bool = False) -> dict:
    samples = build_eval_set()
    r = Retriever()

    n = len(samples)
    hit_at: dict[int, int] = {1: 0, 3: 0, 5: 0}
    mrr_sum = 0.0
    mat_ok = 0
    pos_ok = 0
    route_only_mat_ok = 0
    by_material: Counter = Counter()
    by_material_hit: Counter = Counter()
    misses: list[Sample] = []

    for s in samples:
        hits, decision = r.search(s.question, k=k)
        ids = [h.id for h in hits]

        rank = ids.index(s.qid) + 1 if s.qid in ids else 0
        if rank:
            mrr_sum += 1.0 / rank
            for kk in hit_at:
                if rank <= kk:
                    hit_at[kk] += 1
            by_material_hit[s.material] += 1
        else:
            misses.append(s)
        by_material[s.material] += 1

        # 라우팅 정확도 (gold = qa 청크 자신의 메타)
        # posture 도메인 자세 qa(material=posture)는 정답 material=posture
        if decision.material == s.material:
            mat_ok += 1
            route_only_mat_ok += 1
        if decision.position == s.position:
            pos_ok += 1

        if verbose:
            mark = "✓" if rank else "✗"
            print(f"{mark} rank={rank or '-':>2}  route={decision.material}/{decision.position}  gold={s.material}/{s.position}")
            print(f"   Q: {s.question}")

    results = {
        "n": n,
        "recall@1": hit_at[1] / n,
        "recall@3": hit_at[3] / n,
        "recall@5": hit_at[5] / n,
        "mrr@5": mrr_sum / n,
        "routing_material_acc": mat_ok / n,
        "routing_position_acc": pos_ok / n,
        "by_material_recall@k": {m: by_material_hit[m] / by_material[m] for m in by_material},
        "n_misses": len(misses),
    }
    return results, misses


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--show-misses", action="store_true")
    args = ap.parse_args()

    results, misses = evaluate(k=args.k, verbose=args.verbose)

    print(f"\n평가셋 크기: {results['n']}")
    print(f"Recall@1 : {results['recall@1']:.3f}")
    print(f"Recall@3 : {results['recall@3']:.3f}")
    print(f"Recall@5 : {results['recall@5']:.3f}")
    print(f"MRR@5    : {results['mrr@5']:.3f}")
    print(f"라우팅 material acc: {results['routing_material_acc']:.3f}")
    print(f"라우팅 position acc: {results['routing_position_acc']:.3f}")
    print("\n재질별 Recall@k:")
    for m, v in results["by_material_recall@k"].items():
        print(f"  {m:13s} {v:.3f}")

    if args.show_misses and misses:
        print(f"\nTop-{args.k}에 못 들어온 질문 ({len(misses)}건):")
        for s in misses:
            print(f"  [{s.material}/{s.position}] {s.question}")


if __name__ == "__main__":
    main()
