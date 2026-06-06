# 영상 파일 (별도 공유)

GitHub 용량 제한 때문에 원본 mp4 14개(약 1.3GB)는 이 리포지토리에 포함되지 않습니다.

## 받는 법

팀 공유 드라이브에서 다운로드 후 이 폴더에 그대로 넣으세요.

**파일명은 `dataset/sources.json`의 `video` 필드와 정확히 일치해야 합니다.** 예:

```
dataset/video/stainless steel 6G pipe TIG welding.mp4
dataset/video/02-6G-시럼이란-모재높이설정.mp4
dataset/video/6G최종자세2-높은자세.mp4
```

## 확인

```bash
cd rag_pipeline
./rag videos    # 영상 카탈로그
# 또는 서버 띄우고 브라우저:
./rag serve
# → http://localhost:8000/api/video/video_ss_6g  (재생되면 정상)
```

영상이 없어도 RAG 기능(텍스트 노하우 추출, Agent 피드백)은 정상 동작합니다.
`/api/video/{id}` 엔드포인트만 404를 반환합니다.
