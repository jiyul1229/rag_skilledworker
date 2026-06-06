"""FastAPI 서버 — 프론트엔드용 RAG HTTP API.

실행:
    uvicorn server:app --reload --port 8000
또는:
    python server.py

엔드포인트:
    GET  /api/health                       헬스 체크
    GET  /api/materials                    재질 목록
    GET  /api/positions                    자세 목록
    POST /api/knowhow                      메인: (재질, 자세[, 쿼리]) → 노하우 JSON
    GET  /api/video/{source_id}            영상 파일 스트리밍
    GET  /api/sources                      등록된 모든 영상 메타 리스트
    POST /api/answer                       (데모) 자유 텍스트 → Gemini 답변

자동 문서:
    Swagger UI:  http://localhost:8000/docs
    ReDoc:       http://localhost:8000/redoc
"""
from __future__ import annotations

import random
import shutil
import uuid
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import agent
import chatbot
import citations
import rag_api
from loader import DATASET_DIR


UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


app = FastAPI(
    title="Welding RAG API",
    description="파이프 TIG 용접 RAG — 분류된 (재질, 자세)를 받아 노하우 JSON 반환.",
    version="1.0.0",
)

# 프론트 개발 편의: 모든 origin 허용. 배포 시 도메인 좁히세요.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ───────── 요청/응답 스키마 ─────────

Material = Literal["carbon_steel", "stainless", "aluminum"]
Position = Literal["1G", "2G", "5G", "6G"]


class KnowhowRequest(BaseModel):
    material: Material = Field(..., description="분류된 파이프 재질")
    position: Position = Field(..., description="분류된 작업 자세")
    query: Optional[str] = Field(None, description="사용자 자유 질문 (선택, 청크 순위 재조정용)")
    top_k: int = Field(5, ge=1, le=20, description="쿼리 모드에서 섹션별 상위 N")
    include_posture: Optional[bool] = Field(
        None, description="자세 가이드 포함 여부. None이면 6G일 때만 자동 포함."
    )


class AnswerRequest(BaseModel):
    query: str = Field(..., description="자유 텍스트 질문")
    k: int = Field(5, ge=1, le=10)
    dry_run: bool = Field(False, description="True면 LLM 호출 없이 프롬프트만 반환")


class FeedbackRequest(BaseModel):
    material: Material
    position: Position
    observation: Optional[str] = Field(
        None, description="사용자가 묘사한 현재 작업 상황/문제 (예: '백비드가 검게 나와요')"
    )
    top_k: int = Field(5, ge=1, le=10)
    dry_run: bool = Field(False)


# ── 가상 분류기 ──
# 계획서: "분류 모델 직접 개발 x → 가상 시나리오로 가정"
# 폼에 material/position이 오면 그대로, 없으면 파일명 힌트로 추정, 그것도 없으면 랜덤.
_MATERIAL_HINTS = {
    "carbon": "carbon_steel", "탄소": "carbon_steel", "cs": "carbon_steel",
    "stainless": "stainless", "스테인": "stainless", "스텐": "stainless", "sus": "stainless", "ss": "stainless",
    "aluminum": "aluminum", "aluminium": "aluminum", "알루미": "aluminum",
}
_POSITION_HINTS = ["1G", "2G", "5G", "6G"]


def _fake_classify(filename: str, mat: Optional[str], pos: Optional[str]) -> dict:
    if mat in rag_api.VALID_MATERIALS and pos in rag_api.VALID_POSITIONS:
        return {"material": mat, "position": pos, "confidence": 1.0, "source": "form"}

    low = filename.lower()
    found_mat = None
    for k, v in _MATERIAL_HINTS.items():
        if k in low:
            found_mat = v
            break
    found_pos = None
    for p in _POSITION_HINTS:
        if p.lower() in low:
            found_pos = p
            break

    if found_mat and found_pos:
        return {"material": found_mat, "position": found_pos,
                "confidence": 0.85, "source": "filename"}

    # 마지막 fallback: 의도적 가짜 분류
    rng = random.Random(filename)
    return {
        "material": found_mat or rng.choice(sorted(rag_api.VALID_MATERIALS)),
        "position": found_pos or rng.choice(_POSITION_HINTS),
        "confidence": 0.5,
        "source": "random_fallback",
    }


# ───────── 엔드포인트 ─────────

@app.get("/api/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/materials", tags=["meta"])
def list_materials() -> list[str]:
    return sorted(rag_api.VALID_MATERIALS)


@app.get("/api/positions", tags=["meta"])
def list_positions() -> list[str]:
    return sorted(rag_api.VALID_POSITIONS)


@app.post("/api/knowhow", tags=["rag"])
def knowhow(req: KnowhowRequest) -> dict[str, Any]:
    """RAG의 메인 출구. (재질, 자세) → 구조화된 노하우 JSON.

    응답 키:
    - parameters: 표준 작업 파라미터 (전류/가스/텅스텐 등)
    - expert_tips: [{stage, tip}]
    - defect_solutions: [{defect, cause, solution}]
    - qa: [{question, answer}]
    - guide_sections: [{title, body}]
    - posture_notes: 6G일 때만 (재질 무관 공통 자세 가이드)
    - citations: [{id, title, video, material, position}]
    - missing_videos: [str]
    """
    try:
        return rag_api.get_knowhow(
            material=req.material,
            position=req.position,
            query=req.query,
            top_k=req.top_k,
            include_posture=req.include_posture,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/sources", tags=["meta"])
def list_sources() -> list[dict]:
    """등록된 영상 메타 전체. 프론트가 영상 라이브러리 만들 때 유용."""
    by_id, _ = citations._load_sources()
    return [
        {"id": c.id, "title": c.title, "video_url": f"/api/video/{c.id}",
         "material": c.material, "position": c.position}
        for c in by_id.values()
    ]


@app.get("/api/video/{source_id}", tags=["video"])
def get_video(source_id: str) -> FileResponse:
    """source_id로 영상 파일 스트리밍 (Range 요청 지원)."""
    cits = citations.resolve([source_id])
    if not cits:
        raise HTTPException(status_code=404, detail=f"unknown source_id: {source_id}")
    rel = cits[0].video
    if not rel:
        raise HTTPException(status_code=404, detail="video path missing")
    path = DATASET_DIR / rel
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"video file not found on disk: {rel}",
        )
    return FileResponse(path, media_type="video/mp4", filename=path.name)


@app.post("/api/answer", tags=["demo"])
def answer(req: AnswerRequest) -> dict[str, Any]:
    """(데모) 자유 텍스트 → 검색 → Gemini 답변 + 인용.

    실제 프로덕션 흐름은: 프론트 → /api/knowhow → Agent 서비스 → 답변.
    이 엔드포인트는 Agent가 아직 없을 때의 임시 데모/MVP용.
    """
    a = chatbot.answer(req.query, k=req.k, dry_run=req.dry_run)
    return {
        "answer": a.text,
        "routing": {
            "material": a.decision.material,
            "position": a.decision.position,
            "reason": a.decision.reason,
        },
        "citations_markdown": a.citations_md,
        "hits": [
            {
                "id": h.id, "score": h.score, "material": h.material,
                "position": h.position, "type": h.type,
                "stage": h.stage, "defect": h.defect,
                "text": h.text, "source_ids": h.source_ids,
                "source_file": h.source_file,
            } for h in a.hits
        ],
    }


@app.get("/api/training-videos", tags=["video"])
def training_videos(
    material: Optional[Material] = None,
    position: Optional[Position] = None,
) -> list[dict]:
    """사전교육용 영상 라이브러리. 재질/자세로 필터 가능.

    프론트는 이 목록을 카드 그리드로 그려서, 클릭하면 /api/video/{id}로 재생.
    """
    by_id, _ = citations._load_sources()
    out: list[dict] = []
    for c in by_id.values():
        if material and c.material and c.material != material:
            continue
        if position and c.position and c.position != position:
            continue
        out.append({
            "id": c.id, "title": c.title,
            "material": c.material, "position": c.position,
            "video_url": f"/api/video/{c.id}",
        })
    return out


@app.post("/api/upload", tags=["feedback"])
def upload(
    file: UploadFile = File(..., description="작업 사진 또는 영상"),
    material: Optional[str] = Form(None, description="(선택) 정답 강제. 가상 분류 대신 사용"),
    position: Optional[str] = Form(None, description="(선택) 정답 강제"),
) -> dict:
    """파일 업로드 + 가상 분류.

    실제 분류 모델은 본 프로젝트 범위 밖. 폼 값이 있으면 그걸 쓰고,
    없으면 파일명 힌트 / 랜덤 fallback. 응답의 confidence/source로 어떻게 결정됐는지 표기.
    """
    if not file.filename:
        raise HTTPException(400, "file required")
    upload_id = uuid.uuid4().hex[:12]
    suffix = Path(file.filename).suffix
    dest = UPLOAD_DIR / f"{upload_id}{suffix}"
    with dest.open("wb") as fp:
        shutil.copyfileobj(file.file, fp)

    classification = _fake_classify(file.filename, material, position)
    return {
        "upload_id": upload_id,
        "stored_path": str(dest.relative_to(Path(__file__).resolve().parent)),
        "original_filename": file.filename,
        "size_bytes": dest.stat().st_size,
        "classification": classification,
    }


@app.post("/api/feedback", tags=["feedback"])
def feedback(req: FeedbackRequest) -> dict:
    """분류된 (재질, 자세) + 사용자 관찰 → 노하우 + 영상 + Agent 피드백.

    실시간 피드백의 통합 출구. 응답은 그대로 프론트가 렌더링하면 됨:
      - feedback.summary / key_points / warnings / next_steps
      - training_videos: 같은 (재질, 자세)의 사전교육 영상 카드
      - citations: 인용 영상 (피드백 근거)
      - knowhow 요약 통계
    """
    try:
        result = agent.generate_feedback(
            material=req.material,
            position=req.position,
            observation=req.observation,
            top_k=req.top_k,
            dry_run=req.dry_run,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    # 같은 분류의 사전교육 영상 묶음 (이미 인용된 것 제외)
    cit_ids = {c["id"] for c in result.get("citations", [])}
    by_id, _ = citations._load_sources()
    training = [
        {"id": c.id, "title": c.title, "material": c.material,
         "position": c.position, "video_url": f"/api/video/{c.id}"}
        for c in by_id.values()
        if c.id not in cit_ids
        and (not c.material or c.material == req.material)
        and (not c.position or c.position == req.position)
    ]
    result["training_videos"] = training
    return result


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {
        "name": "Welding RAG API",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
