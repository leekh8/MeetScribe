# MeetScribe

로컬에서 완결되는 회의 녹음 → 마크다운 회의록 자동화. 녹음이 기기 밖으로 나가지 않는 **완전 오프라인 STT**를 지향한다.

클라우드 STT(클로바·구글 등)와 달리 오디오를 외부로 전송하지 않으므로, 민감한 회의 녹음을 다룰 때 프라이버시가 보장된다.

## 파이프라인

```
[오디오 m4a/wav/mp3]
      │  transcribe  (faster-whisper, 로컬 CPU/GPU)
      ▼
   세그먼트(타임스탬프 + 텍스트)
      │  corrections (도메인 용어 후처리 교정)
      ▼
   교정된 세그먼트
      │  diarize     (pyannote, 선택) ─ 화자 태깅
      ▼
   화자별 세그먼트
      │  summarize   (LLM, 선택) ─ 요약·결정·액션아이템
      ▼
   render → 마크다운 회의록 (.md)
```

전사·교정·렌더는 의존성만 있으면 완전 오프라인으로 동작한다. 화자 분리(diarize)와 요약(summarize)은 선택 단계다.

## 설치

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# ffmpeg 필요 (winget install ffmpeg 또는 apt install ffmpeg)
```

Whisper 모델은 첫 실행 시 자동 다운로드된다(medium ≈ 1.5GB, large-v3 ≈ 3GB).

## 사용법

```bash
# 단일 파일 → 회의록 md
python cli.py "회의녹음.m4a"

# 모델 지정 (기본 medium)
python cli.py "회의녹음.m4a" --model large-v3

# 출력 위치
python cli.py "회의녹음.m4a" -o out/

# 화자 분리 포함 (Phase 2, pyannote + HF 토큰 필요)
python cli.py "회의녹음.m4a" --diarize

# LLM 요약 포함 (Phase 3)
python cli.py "회의녹음.m4a" --summarize
```

## 도메인 용어 사전

오인식되기 쉬운 IT/보안 용어를 후처리로 교정한다. 기본 사전은 `meetscribe/corrections.py`에 있고,
`corrections.local.json`(gitignore됨)을 두면 개인·조직 특화 용어를 추가할 수 있다.

```json
{ "시엠": "SIEM", "씨브이이": "CVE", "우리회사제품명": "정확한표기" }
```

## 개발 상태

- [x] Phase 0 — 모듈 리팩터·스캐폴딩
- [ ] Phase 1 — 오프라인 파이프라인 end-to-end 검증
- [ ] Phase 2 — 화자 분리(pyannote)
- [ ] Phase 3 — LLM 요약·액션아이템
- [ ] Phase 4 — Streamlit UI

## 라이선스

MIT (예정)
