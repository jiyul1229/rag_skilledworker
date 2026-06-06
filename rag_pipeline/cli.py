"""통합 CLI — 한 명령어로 모든 RAG 기능 호출.

사용:
    python cli.py <command> [options]

예시:
    python cli.py knowhow stainless 6G --query "백비드 검게"
    python cli.py feedback stainless 6G "백비드가 검게 나와요"
    python cli.py videos stainless 6G
    python cli.py search "탄소강 5G 캡 패스 언더컷"
    python cli.py classify /tmp/stainless_6G.jpg
    python cli.py chat "스테인 6G 백비드 검게"
    python cli.py eval
    python cli.py serve
    python cli.py build
    python cli.py help

기본 출력은 사람이 읽기 쉬운 형식. --json 옵션 주면 raw JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ───────── pretty printers ─────────

def _hr(c: str = "─", w: int = 70) -> None:
    print(c * w)


def _title(s: str) -> None:
    _hr("═")
    print(f"  {s}")
    _hr("═")


def _section(s: str) -> None:
    print()
    print(f"▼ {s}")
    _hr()


def _bullet(s: str, indent: int = 0) -> None:
    print(" " * indent + "• " + s)


def _kv(k: str, v, indent: int = 0) -> None:
    print(" " * indent + f"{k}: {v}")


# ───────── commands ─────────

def cmd_knowhow(args: argparse.Namespace) -> None:
    import rag_api
    data = rag_api.get_knowhow(
        material=args.material, position=args.position,
        query=args.query, top_k=args.top_k,
    )
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return

    _title(f"노하우 [{data['material']} / {data['position']}]"
           + (f"  ← \"{data['query']}\"" if data["query"] else ""))

    if data.get("parameters"):
        _section("표준 파라미터")
        for k, v in data["parameters"].items():
            _kv(k, v)

    if data.get("expert_tips"):
        _section(f"숙련공 팁 ({len(data['expert_tips'])})")
        for t in data["expert_tips"]:
            stage = f"[{t['stage']}] " if t.get("stage") else ""
            _bullet(f"{stage}{t['tip']}")

    if data.get("defect_solutions"):
        _section(f"결함 해결 ({len(data['defect_solutions'])})")
        for d in data["defect_solutions"]:
            print(f"  ◇ {d['defect']}")
            print(f"    원인: {d['cause']}")
            print(f"    해결: {d['solution']}")

    if data.get("qa"):
        _section(f"Q&A ({len(data['qa'])})")
        for q in data["qa"]:
            print(f"  Q. {q['question']}")
            print(f"  A. {q['answer']}\n")

    if data.get("guide_sections"):
        _section(f"가이드 문서 ({len(data['guide_sections'])} 섹션)")
        for g in data["guide_sections"]:
            print(f"  ◇ {g['title']}")
            preview = (g["body"] or "").replace("\n", " ")[:120]
            print(f"    {preview}…\n")

    if data.get("posture_notes"):
        _section(f"자세 가이드 ({len(data['posture_notes'])} 항목, 재질 무관 6G 공통)")
        for n in data["posture_notes"][:5]:
            sub = f"[{n['subtopic']}] " if n.get("subtopic") else ""
            tip = n.get("tip") or n.get("solution") or n.get("answer") or ""
            _bullet(f"{sub}{tip[:100]}")
        if len(data["posture_notes"]) > 5:
            print(f"  … 외 {len(data['posture_notes']) - 5}개")

    if data.get("citations"):
        _section(f"인용 영상 ({len(data['citations'])})")
        for c in data["citations"]:
            _bullet(f"{c['title']}  ({c['video']})")


def cmd_feedback(args: argparse.Namespace) -> None:
    import agent
    out = agent.generate_feedback(
        material=args.material, position=args.position,
        observation=args.observation, dry_run=args.dry_run,
    )
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    fb = out["feedback"]
    _title(f"Agent 피드백 [{args.material} / {args.position}]")
    if args.observation:
        _kv("상황 설명", args.observation)
    _kv("LLM", f"{out['llm'].get('provider','?')}/{out['llm']['model']} (env={out['llm']['used_env']}, dry_run={out['llm']['dry_run']})")

    _section("진단 요약")
    print(f"  {fb.get('summary', '')}")
    if fb.get("key_points"):
        _section("핵심 점검 사항")
        for x in fb["key_points"]:
            _bullet(x)
    if fb.get("warnings"):
        _section("⚠ 주의")
        for x in fb["warnings"]:
            _bullet(x)
    if fb.get("next_steps"):
        _section("다음 단계")
        for i, x in enumerate(fb["next_steps"], 1):
            print(f"  {i}. {x}")

    cits = out.get("citations", [])
    if cits:
        _section(f"근거 영상 ({len(cits)})")
        for c in cits:
            _bullet(f"{c['title']}  ({c['video']})")


def cmd_videos(args: argparse.Namespace) -> None:
    import citations
    by_id, missing = citations._load_sources()
    rows = []
    for c in by_id.values():
        if args.material and c.material and c.material != args.material:
            continue
        if args.position and c.position and c.position != args.position:
            continue
        rows.append(c)
    if args.json:
        print(json.dumps([{
            "id": c.id, "title": c.title, "video": c.video,
            "material": c.material, "position": c.position,
        } for c in rows], ensure_ascii=False, indent=2))
        return

    flt = []
    if args.material: flt.append(args.material)
    if args.position: flt.append(args.position)
    _title("영상 카탈로그" + (f" [{' / '.join(flt)}]" if flt else ""))
    print(f"  매칭: {len(rows)} / 전체: {len(by_id)}\n")
    for c in rows:
        tag = f"[{c.material or '-'}/{c.position or '-'}]"
        print(f"  {c.id:25s} {tag:18s} {c.title}")
        print(f"  {'':25s} {'':18s} → {c.video}\n")
    if missing:
        _section(f"미수집 ({len(missing)})")
        for m in missing:
            _bullet(m)


def cmd_search(args: argparse.Namespace) -> None:
    from retriever import Retriever
    r = Retriever()
    hits, decision = r.search(args.query, k=args.k,
                              material=args.material, position=args.position)
    if args.json:
        print(json.dumps({
            "query": args.query,
            "routing": {"material": decision.material, "position": decision.position,
                        "reason": decision.reason},
            "hits": [{
                "id": h.id, "score": h.score, "material": h.material,
                "position": h.position, "type": h.type, "stage": h.stage,
                "defect": h.defect, "text": h.text,
            } for h in hits],
        }, ensure_ascii=False, indent=2))
        return

    _title(f"검색: \"{args.query}\"")
    _kv("라우팅", decision.reason)
    print()
    for i, h in enumerate(hits, 1):
        head = f"[{i}] score={h.score:.3f}  {h.material}/{h.position}/{h.type}"
        if h.stage: head += f" · {h.stage}"
        if h.defect: head += f" · defect={h.defect}"
        print(head)
        for line in h.text.splitlines():
            print(f"    {line}")
        print()


def cmd_classify(args: argparse.Namespace) -> None:
    # server._fake_classify와 동일 로직을 CLI에서 호출
    import server  # 로직 재사용
    cls = server._fake_classify(args.file, args.material, args.position)
    if args.json:
        print(json.dumps({"file": args.file, "classification": cls}, ensure_ascii=False, indent=2))
        return
    _title("가상 분류")
    _kv("파일", args.file)
    _kv("재질", cls["material"])
    _kv("자세", cls["position"])
    _kv("신뢰도", cls["confidence"])
    _kv("판정 근거", cls["source"])
    print()
    print("  → 다음 단계 예시:")
    print(f"    python cli.py feedback {cls['material']} {cls['position']} \"현재 상황\"")


def cmd_chat(args: argparse.Namespace) -> None:
    import chatbot
    a = chatbot.answer(args.query, k=args.k, dry_run=args.dry_run)
    if args.json:
        print(json.dumps({
            "query": args.query,
            "answer": a.text,
            "routing": {"material": a.decision.material, "position": a.decision.position},
            "citations_markdown": a.citations_md,
        }, ensure_ascii=False, indent=2))
        return
    _title(f"챗봇: \"{args.query}\"")
    _kv("라우팅", a.decision.reason)
    _section("답변")
    print(a.text)
    _section("참고 영상")
    print(a.citations_md)


def cmd_eval(args: argparse.Namespace) -> None:
    import evaluate
    r, misses = evaluate.evaluate(k=args.k, verbose=False)
    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return
    _title("RAG 검색 품질 평가")
    _kv("평가셋 크기", r["n"])
    _kv("Recall@1", f"{r['recall@1']:.3f}")
    _kv("Recall@3", f"{r['recall@3']:.3f}")
    _kv("Recall@5", f"{r['recall@5']:.3f}")
    _kv("MRR@5",    f"{r['mrr@5']:.3f}")
    _kv("라우팅 material acc", f"{r['routing_material_acc']:.3f}")
    _kv("라우팅 position acc", f"{r['routing_position_acc']:.3f}")
    _section("재질별 Recall@k")
    for m, v in r["by_material_recall@k"].items():
        _kv(m, f"{v:.3f}", indent=2)
    if args.show_misses and misses:
        _section(f"Top-{args.k}에 못 들어온 질문 ({len(misses)}건)")
        for s in misses:
            _bullet(f"[{s.material}/{s.position}] {s.question}")


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn
    print(f"▶ uvicorn server:app  host={args.host}  port={args.port}")
    print(f"  Swagger:  http://localhost:{args.port}/docs")
    print(f"  ReDoc:    http://localhost:{args.port}/redoc")
    print(f"  Health:   http://localhost:{args.port}/api/health")
    print(f"  (이 터미널에서 Ctrl+C로 중지)")
    uvicorn.run("server:app", host=args.host, port=args.port, reload=args.reload)


def cmd_build(args: argparse.Namespace) -> None:
    import build_index
    build_index.build()
    if args.test:
        build_index.smoke_test()


def cmd_help(args: argparse.Namespace) -> None:
    print(__doc__)
    print("\n전체 서브커맨드:")
    print("  knowhow   (재질, 자세[, 쿼리]) → 노하우 JSON/표")
    print("  feedback  분류 + 관찰 → Agent 피드백")
    print("  videos    영상 목록 (재질/자세 필터)")
    print("  search    자유 텍스트 → 청크 검색")
    print("  classify  파일 → 가상 분류 결과")
    print("  chat      자유 텍스트 → Gemini 답변")
    print("  eval      Recall@k / MRR 측정")
    print("  serve     FastAPI 서버 띄우기")
    print("  build     Chroma 인덱스 재빌드")


# ───────── arg parsing ─────────

MATERIAL_CHOICES = ["carbon_steel", "stainless", "aluminum"]
POSITION_CHOICES = ["1G", "2G", "5G", "6G"]


def _add_json(p: argparse.ArgumentParser) -> None:
    p.add_argument("--json", action="store_true", help="raw JSON 출력")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rag",
        description="파이프 TIG 용접 RAG 통합 CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # knowhow
    a = sub.add_parser("knowhow", help="(재질, 자세) → 노하우")
    a.add_argument("material", choices=MATERIAL_CHOICES)
    a.add_argument("position", choices=POSITION_CHOICES)
    a.add_argument("--query", help="자유 텍스트 쿼리 (청크 순위 재조정)")
    a.add_argument("--top-k", type=int, default=5)
    _add_json(a); a.set_defaults(func=cmd_knowhow)

    # feedback
    a = sub.add_parser("feedback", help="Agent 피드백 생성")
    a.add_argument("material", choices=MATERIAL_CHOICES)
    a.add_argument("position", choices=POSITION_CHOICES)
    a.add_argument("observation", nargs="?", default=None, help="현 상황 설명 (선택)")
    a.add_argument("--dry-run", action="store_true")
    _add_json(a); a.set_defaults(func=cmd_feedback)

    # videos
    a = sub.add_parser("videos", help="영상 카탈로그")
    a.add_argument("material", nargs="?", choices=MATERIAL_CHOICES, default=None)
    a.add_argument("position", nargs="?", choices=POSITION_CHOICES, default=None)
    _add_json(a); a.set_defaults(func=cmd_videos)

    # search
    a = sub.add_parser("search", help="자유 텍스트 검색")
    a.add_argument("query")
    a.add_argument("-k", type=int, default=5)
    a.add_argument("--material", choices=MATERIAL_CHOICES)
    a.add_argument("--position", choices=POSITION_CHOICES)
    _add_json(a); a.set_defaults(func=cmd_search)

    # classify
    a = sub.add_parser("classify", help="파일 → 가상 분류")
    a.add_argument("file")
    a.add_argument("--material", help="강제 지정 (가짜 분류 우회)")
    a.add_argument("--position", help="강제 지정 (가짜 분류 우회)")
    _add_json(a); a.set_defaults(func=cmd_classify)

    # chat
    a = sub.add_parser("chat", help="Gemini 챗봇")
    a.add_argument("query")
    a.add_argument("-k", type=int, default=5)
    a.add_argument("--dry-run", action="store_true")
    _add_json(a); a.set_defaults(func=cmd_chat)

    # eval
    a = sub.add_parser("eval", help="검색 품질 평가")
    a.add_argument("-k", type=int, default=5)
    a.add_argument("--show-misses", action="store_true")
    _add_json(a); a.set_defaults(func=cmd_eval)

    # serve
    a = sub.add_parser("serve", help="FastAPI 서버 띄우기 (WSL/원격에서도 접근 가능)")
    a.add_argument("--host", default="0.0.0.0", help="기본 0.0.0.0 — WSL2 윈도우 브라우저 호환")
    a.add_argument("--port", type=int, default=8000)
    a.add_argument("--reload", action="store_true")
    a.set_defaults(func=cmd_serve)

    # build
    a = sub.add_parser("build", help="Chroma 인덱스 빌드")
    a.add_argument("--test", action="store_true")
    a.set_defaults(func=cmd_build)

    # help
    a = sub.add_parser("help", help="자세한 도움말")
    a.set_defaults(func=cmd_help)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
