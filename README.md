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

# LLM 요약 포함 (Phase 3 — 현재 미구현, 실행 시 경고 후 생략)
python cli.py "회의녹음.m4a" --summarize

# 언어 지정 / 기존 출력 덮어쓰기 / 조용히
python cli.py "meeting.m4a" --language en --force --quiet

# 전사 없이 입력·출력 경로만 미리보기
python cli.py "회의녹음.m4a" --dry-run
```

### 주요 옵션

| 옵션 | 설명 |
|------|------|
| `--model {tiny,base,small,medium,large-v3}` | Whisper 모델 (기본 `medium`) |
| `--language <code>` | 전사 언어 (기본 `ko`, 예: `en`/`ja`) |
| `-o, --out-dir <dir>` | 출력 디렉토리 |
| `--diarize` | 화자 분리 (pyannote 미설치 시 경고 후 생략) |
| `--summarize` | LLM 요약 (Phase 3 미구현 — 경고 후 생략) |
| `--hf-token <token>` | 화자 분리용 HF 토큰 (또는 `HF_TOKEN` 환경변수) |
| `--force` | 기존 `{파일명}.md` 덮어쓰기 (기본은 거부) |
| `--dry-run` | 전사 없이 경로·설정만 출력 |
| `--quiet` | 진행 로그 숨김 |
| `--version` | 버전 출력 |

> 지원 입력: `.m4a .wav .mp3 .flac .ogg .aac .mp4 .webm .opus`. 그 외 확장자·없는 파일은 전사 시작 전에 즉시 오류로 걸러진다.

## 도메인 용어 사전

오인식되기 쉬운 IT/보안 용어를 후처리로 교정한다. 기본 사전은 `meetscribe/config.py`에 있고,
`corrections.local.json`(gitignore됨)을 두면 개인/조직 특화 용어를 추가할 수 있다.

```json
{ "시엠": "SIEM", "씨브이이": "CVE", "우리회사제품명": "정확한표기" }
```

영문/숫자로만 된 키에는 단어 경계가 자동으로 붙는다. `"UR": "URL"` 규칙이 `URL`을 `URLL`로,
`during`을 `dURLing`으로 만들지 않는다. 한글 키는 조사가 붙어 오므로 경계를 걸지 않는다.

### 사전을 손으로 채우지 않는 법

전사본을 처음부터 읽으면서 고치는 것은 현실적으로 유지되지 않는다.
대신 오인식으로 보이는 후보만 뽑아 질문지로 만들고, 답만 받아 사전에 넣는다.

```bash
# 전사본(md/json)이 쌓인 디렉토리에서 후보 추출
python cli.py --suggest-terms path/to/transcripts

#  -> out/terms_to_ask.md   사람이 읽을 질문지
#  -> out/terms_to_ask.json 기계 판독용

# 답을 {"오인식": "정답"} 형태로 적어 사전에 반영
python cli.py --apply-terms answers.json
```

후보는 모델 없이 세 신호로 찾는다.

| 신호 | 잡아내는 것 | 예 |
|------|-------------|-----|
| `variant` | 같은 고유명사를 매번 다르게 알아들은 흔적 | 스프랑크 / 스프렁크 / 스프랑커 |
| `domain` | 특정 회의에만 몰려 나오는 토큰(tf-idf) | 워팔라이저, 유지보스 |
| `latin` | 영문 표기 흔들림 | fortisoar / FortiSOAR |

`variant`는 첫 음절이 같고 한 음절만 다른 3음절 이상 토큰만 묶는다.
단순 편집거리로는 한국어 어절이 촘촘해 `하고 / 가고 / 갖고 / 같고`가 한 묶음이 되어 쓸 수 없다.
용언 활용형(`확인해 / 확인하`)도 마지막 음절이 어미면 제외한다.

사전이 커질수록 후보는 줄어든다. 회의 7건 기준으로 한 바퀴 돌리면 `variant` 묶음이 거의 소진된다.

## 테스트

순수 함수(용어 교정, 회의록 렌더, 타임스탬프, 화자 배정, 사전 로딩, 후보 발굴)를 pytest로 검증한다.

```bash
pip install -r requirements-dev.txt
python -m pytest              # 47 tests
```

전사(faster-whisper)·화자 분리(pyannote)처럼 대용량 모델·외부 실행이 필요한 부분은
지연 import로 분리돼 있어, 테스트는 모델 없이 로직만 빠르게 돈다.

## 개발 상태

- [x] Phase 0 — 모듈 리팩터·스캐폴딩
- [ ] Phase 1 — 오프라인 파이프라인 end-to-end 검증 (순수 로직 pytest 커버 완료)
- [ ] Phase 2 — 화자 분리(pyannote)
- [ ] Phase 3 — LLM 요약, 액션아이템 (로컬 LLM 백엔드 검토 중)
- [ ] Phase 4 — Streamlit UI

## 라이선스

MIT (예정)
