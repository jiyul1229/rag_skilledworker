"""Agent — 노하우 JSON을 받아 사용자 피드백 report를 만든다.

chatbot.py는 자유 Q&A용이고, 이 모듈은 "분류 결과 + 노하우" 컨텍스트에서
신입에게 코칭하듯 구조화된 피드백을 생성한다.

흐름:
  (material, position) + 사용자 상황 설명(observation)
    → rag_api.get_knowhow()
    → 피드백 프롬프트로 Gemini 호출
    → 구조화된 dict: {summary, key_points, warnings, next_steps, citations}

JSON 출력을 강제하기 위해 response_schema 사용.
"""
from __future__ import annotations

import json
import os
from typing import Any

import rag_api
from chatbot import _resolve_provider


SYSTEM_PROMPT = """너는 30년 경력 파이프 TIG 용접 숙련공이다. 신입 작업자가 자신의 작업 상황을 설명하면, 제공된 <knowhow> JSON을 근거로 즉시 적용 가능한 피드백을 만든다.

규칙:
- <knowhow>의 parameters / expert_tips / defect_solutions / qa / guide_sections / posture_notes만 근거로 사용한다. 외부 지식 금지.
- 숫자(전류, 가스 LPM, 갭, 각도)는 knowhow에 있는 값을 그대로 인용한다.
- 답은 한국어. 현장에서 후배에게 말하듯 짧고 직선적으로.
- 출력은 반드시 다음 키를 가진 JSON:
  summary    : 1~2문장. 현 상황 진단 요약.
  key_points : 3~5개 bullet. 지금 당장 점검/조정할 것.
  warnings   : 0~3개 bullet. 위험하거나 자주 망치는 함정.
  next_steps : 2~4개 bullet. 다음에 시도할 순서.
- knowhow에 답이 없는 항목은 빈 배열로 둔다. 절대 추측해서 채우지 마라.
"""


FEEDBACK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "key_points": {"type": "array", "items": {"type": "string"}},
        "warnings": {"type": "array", "items": {"type": "string"}},
        "next_steps": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "key_points", "warnings", "next_steps"],
}


def _format_knowhow_for_prompt(kh: dict) -> str:
    """LLM에게 전달할 컨텍스트 문자열로 정리. 너무 크면 잘라낸다."""
    compact: dict[str, Any] = {
        "material": kh["material"],
        "position": kh["position"],
        "parameters": kh.get("parameters"),
        "expert_tips": kh.get("expert_tips", []),
        "defect_solutions": kh.get("defect_solutions", []),
        "qa": kh.get("qa", []),
        # guide_sections는 길어서 body만 1500자 컷
        "guide_sections": [
            {"title": g["title"], "body": (g["body"] or "")[:1500]}
            for g in kh.get("guide_sections", [])
        ],
        "posture_notes": kh.get("posture_notes"),
    }
    return json.dumps(compact, ensure_ascii=False, indent=2)


def generate_feedback(
    material: str,
    position: str,
    observation: str | None = None,
    top_k: int = 5,
    dry_run: bool = False,
) -> dict[str, Any]:
    """분류 결과 + 사용자 관찰을 받아 구조화된 피드백 반환.

    반환 dict:
      classification : {material, position}
      observation    : 입력한 상황 설명 (없으면 null)
      feedback       : {summary, key_points, warnings, next_steps}
      knowhow        : 원본 RAG JSON 일부 (citations / parameters 포함)
      citations      : 인용 영상 리스트
      llm            : {model, used_env, dry_run}
    """
    knowhow = rag_api.get_knowhow(
        material=material,
        position=position,
        query=observation,
        top_k=top_k,
    )

    api_key, provider, env_used, model = _resolve_provider()

    if dry_run or not api_key:
        feedback = {
            "summary": f"[{('DRY-RUN' if dry_run else '키 미설정')}] 실제 LLM 호출은 생략. 컨텍스트 어셈블만 검증.",
            "key_points": [],
            "warnings": [],
            "next_steps": [],
        }
        llm_info = {
            "provider": provider,
            "model": model,
            "used_env": env_used,
            "dry_run": True,
            "reason": "dry_run" if dry_run else "no_api_key",
        }
    else:
        if provider == "groq":
            feedback = _call_groq_json(
                system=SYSTEM_PROMPT,
                knowhow_str=_format_knowhow_for_prompt(knowhow),
                observation=observation or "",
                api_key=api_key,
                model=model,
            )
        else:
            feedback = _call_gemini_json(
                system=SYSTEM_PROMPT,
                knowhow_str=_format_knowhow_for_prompt(knowhow),
                observation=observation or "",
                api_key=api_key,
                model=model,
            )
        llm_info = {"provider": provider, "model": model, "used_env": env_used, "dry_run": False}

    return {
        "classification": {"material": material, "position": position},
        "observation": observation,
        "feedback": feedback,
        "knowhow": {
            "parameters": knowhow.get("parameters"),
            "guide_sections_count": len(knowhow.get("guide_sections") or []),
            "tips_count": len(knowhow.get("expert_tips") or []),
            "defects_count": len(knowhow.get("defect_solutions") or []),
        },
        "citations": knowhow.get("citations", []),
        "llm": llm_info,
    }


def _build_user_msg(knowhow_str: str, observation: str) -> str:
    return (
        f"<knowhow>\n{knowhow_str}\n</knowhow>\n\n"
        f"<observation>\n{observation or '(사용자가 상황 설명을 제공하지 않음. 일반 가이드 위주로.)'}\n</observation>\n\n"
        "위 knowhow를 근거로 신입에게 줄 피드백 JSON을 생성하라. "
        "반드시 JSON 객체만 반환 (코드 펜스 없이)."
    )


def _normalize_feedback(parsed: dict) -> dict[str, Any]:
    for k in ("summary", "key_points", "warnings", "next_steps"):
        parsed.setdefault(k, [] if k != "summary" else "")
    return parsed


def _call_groq_json(
    system: str,
    knowhow_str: str,
    observation: str,
    api_key: str,
    model: str,
) -> dict[str, Any]:
    try:
        from groq import Groq
    except ImportError as e:
        return {
            "summary": f"[groq SDK 미설치 — `pip install groq`] {e}",
            "key_points": [], "warnings": [], "next_steps": [],
        }

    client = Groq(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": _build_user_msg(knowhow_str, observation)},
            ],
            temperature=0.2,
            max_tokens=2048,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        return {
            "summary": f"[Groq 호출 실패 — model={model}] {type(e).__name__}: {e}",
            "key_points": [], "warnings": [], "next_steps": [],
        }

    raw = (resp.choices[0].message.content or "").strip()
    try:
        return _normalize_feedback(json.loads(raw))
    except json.JSONDecodeError:
        return {
            "summary": "[Groq JSON 파싱 실패]",
            "key_points": [raw[:500]],
            "warnings": [], "next_steps": [],
        }


def _call_gemini_json(
    system: str,
    knowhow_str: str,
    observation: str,
    api_key: str,
    model: str,
) -> dict[str, Any]:
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        return {
            "summary": f"[google-genai SDK 미설치] {e}",
            "key_points": [], "warnings": [], "next_steps": [],
        }

    client = genai.Client(api_key=api_key)
    try:
        resp = client.models.generate_content(
            model=model,
            contents=_build_user_msg(knowhow_str, observation),
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=2048,
                temperature=0.2,
                response_mime_type="application/json",
                response_schema=FEEDBACK_SCHEMA,
            ),
        )
    except Exception as e:
        return {
            "summary": f"[Gemini 호출 실패] {type(e).__name__}: {e}",
            "key_points": [], "warnings": [], "next_steps": [],
        }

    raw = (resp.text or "").strip()
    try:
        return _normalize_feedback(json.loads(raw))
    except json.JSONDecodeError:
        return {
            "summary": "[Gemini JSON 파싱 실패]",
            "key_points": [raw[:500]],
            "warnings": [], "next_steps": [],
        }


# ───────── CLI ─────────
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--material", required=True, choices=sorted(rag_api.VALID_MATERIALS))
    ap.add_argument("--position", required=True, choices=sorted(rag_api.VALID_POSITIONS))
    ap.add_argument("--observation", default=None, help="사용자가 설명하는 현 상황")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out = generate_feedback(
        material=args.material,
        position=args.position,
        observation=args.observation,
        dry_run=args.dry_run,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))
