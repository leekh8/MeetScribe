"""MeetScribe CLI — 오디오 → 마크다운 회의록.

    python cli.py 회의녹음.m4a
    python cli.py 회의녹음.m4a --model large-v3 -o out/
    python cli.py 회의녹음.m4a --diarize --summarize
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

# Windows 콘솔(cp949)에서도 한글이 깨지지 않도록 stdout/stderr를 UTF-8로 재설정.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from meetscribe import __version__
from meetscribe.config import DEFAULT_LANGUAGE, DEFAULT_MODEL, DEFAULT_OUTPUT_DIR
from meetscribe.render import parse_meta, render_markdown
from meetscribe.summarize import DEFAULT_HOST as LLM_HOST
from meetscribe.summarize import DEFAULT_MODEL as LLM_MODEL
from meetscribe.transcribe import transcribe

# faster-whisper/ffmpeg가 디코드할 수 있는 대표 확장자.
SUPPORTED_EXTS = {".m4a", ".wav", ".mp3", ".flac", ".ogg", ".aac", ".mp4", ".webm", ".opus"}


def _module_available(name: str) -> bool:
    """설치 여부 확인. find_spec은 부모 패키지가 없으면 ModuleNotFoundError를 던지므로 흡수한다."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def _fail(msg: str, code: int = 1) -> "None":
    _eprint(f"ERROR: {msg}")
    sys.exit(code)


def _suggest_terms(paths: list[Path], out_dir: Path, limit: int) -> None:
    """전사본에서 교정 후보를 뽑아 질문지(md)와 기계 판독용(json)을 낸다."""
    import json as _json

    from meetscribe.config import load_corrections
    from meetscribe.vocab import find_candidates, load_corpus, render_questions

    missing = [p for p in paths if not p.exists()]
    if missing:
        _fail(f"경로 없음: {', '.join(str(p) for p in missing)}")

    docs = load_corpus(paths)
    if not docs:
        _fail("읽을 전사본이 없습니다 (md 또는 json 필요).")

    candidates = find_candidates(docs, load_corrections(), limit=limit)
    if not candidates:
        print("교정 후보 없음 — 사전이 이미 충분하거나 표본이 작습니다.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "terms_to_ask.md"
    json_path = out_dir / "terms_to_ask.json"
    md_path.write_text(render_questions(candidates, docs), encoding="utf-8")
    json_path.write_text(
        _json.dumps([c.to_dict() for c in candidates], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"전사본 {len(docs)}건에서 후보 {len(candidates)}개")
    print(f"질문지: {md_path}")
    print(f"원자료: {json_path}")


def _apply_terms(answers_path: Path) -> None:
    """답변 JSON을 개인 사전에 병합한다."""
    import json as _json

    from meetscribe.config import LOCAL_DICT_PATH
    from meetscribe.vocab import merge_answers

    if not answers_path.exists():
        _fail(f"파일 없음: {answers_path}")
    try:
        answers = _json.loads(answers_path.read_text(encoding="utf-8"))
    except (_json.JSONDecodeError, OSError) as e:
        _fail(f"답변 파일을 읽을 수 없습니다: {e}")
    if not isinstance(answers, dict):
        _fail('답변 파일은 {"오인식": "정답"} 형태의 객체여야 합니다.')

    added, updated = merge_answers(answers, LOCAL_DICT_PATH)
    print(f"사전 반영: 추가 {added}건 / 갱신 {updated}건 → {LOCAL_DICT_PATH}")


def main():
    parser = argparse.ArgumentParser(description="MeetScribe — 로컬 회의록 자동화")
    parser.add_argument("audio", nargs="?", help="오디오 파일 (m4a/wav/mp3 등)")
    parser.add_argument("--suggest-terms", nargs="+", metavar="PATH", default=None,
                        help="전사본(md/json)에서 교정 후보를 뽑아 질문지 생성")
    parser.add_argument("--apply-terms", metavar="JSON", type=Path, default=None,
                        help='답변 JSON({"오인식":"정답"})을 corrections.local.json에 병합')
    parser.add_argument("--limit", type=int, default=60,
                        help="질문지에 담을 후보 수 (기본 60)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        choices=["tiny", "base", "small", "medium", "large-v3"],
                        help=f"Whisper 모델 (기본 {DEFAULT_MODEL})")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE,
                        help=f"전사 언어 코드 (기본 {DEFAULT_LANGUAGE}, 예: en/ja)")
    parser.add_argument("-o", "--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="출력 디렉토리")
    parser.add_argument("--diarize", action="store_true", help="화자 분리 (Phase 2)")
    parser.add_argument("--summarize", action="store_true",
                        help="로컬 LLM 요약 (Ollama 필요)")
    parser.add_argument("--llm-model", default=LLM_MODEL,
                        help=f"요약에 쓸 로컬 모델 (기본 {LLM_MODEL})")
    parser.add_argument("--llm-host", default=LLM_HOST,
                        help=f"Ollama 주소 (기본 {LLM_HOST})")
    parser.add_argument("--hf-token", default=None,
                        help="화자 분리용 HuggingFace 토큰 (미지정 시 HF_TOKEN 환경변수)")
    parser.add_argument("--force", action="store_true", help="기존 출력 파일 덮어쓰기")
    parser.add_argument("--dry-run", action="store_true",
                        help="전사 없이 입력·출력 경로만 확인")
    parser.add_argument("--quiet", action="store_true", help="진행 로그 숨김")
    parser.add_argument("--version", action="version", version=f"MeetScribe {__version__}")
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    # ── 사전 관리 모드 (오디오 없이 동작) ──────────────────────────────────
    if args.apply_terms:
        _apply_terms(args.apply_terms)
        return
    if args.suggest_terms:
        _suggest_terms([Path(p) for p in args.suggest_terms], args.out_dir, args.limit)
        return

    if not args.audio:
        _fail("오디오 파일을 지정하거나 --suggest-terms / --apply-terms 를 쓰세요.")

    # ── 입력 검증 ──────────────────────────────────────────────────────────
    audio_path = Path(args.audio)
    if not audio_path.exists():
        _fail(f"파일 없음: {audio_path}")
    if not audio_path.is_file():
        _fail(f"파일이 아님: {audio_path}")
    if audio_path.suffix.lower() not in SUPPORTED_EXTS:
        _fail(f"지원하지 않는 형식 '{audio_path.suffix}' — 지원: "
              f"{', '.join(sorted(SUPPORTED_EXTS))}")

    out_path = args.out_dir / f"{audio_path.stem}.md"
    if out_path.exists() and not args.force and not args.dry_run:
        _fail(f"출력 파일이 이미 있음: {out_path} (덮어쓰려면 --force)")

    # ── 선택 기능 사전 점검 (비싼 전사 전에 실패/경고) ──────────────────────
    # 전사는 수 분~수십 분 걸리므로, 쓸 수 없는 옵션은 여기서 걸러 낭비를 막는다.
    do_diarize = args.diarize
    do_summarize = args.summarize
    if do_diarize and not _module_available("pyannote.audio"):
        _eprint("경고: pyannote.audio 미설치 — 화자 분리를 건너뜁니다 "
                "(requirements.txt 주석 해제 후 설치).")
        do_diarize = False
    if do_summarize:
        # 전사는 수십 분이 걸린다. 모델이 없어서 실패할 것을 그 전에 확인한다.
        from meetscribe.summarize import check_backend
        try:
            check_backend(args.llm_host, args.llm_model)
        except RuntimeError as e:
            _eprint(f"경고: 요약을 건너뜁니다 - {e}")
            do_summarize = False

    if args.dry_run:
        print(f"[dry-run] 입력 : {audio_path}")
        print(f"[dry-run] 출력 : {out_path}")
        print(f"[dry-run] 모델 : {args.model} / 언어: {args.language}"
              f" / 화자분리: {do_diarize} / 요약: {do_summarize}")
        return

    progress = not args.quiet

    # ── 전사 ───────────────────────────────────────────────────────────────
    try:
        segments = transcribe(audio_path, model_size=args.model,
                              language=args.language, progress=progress)
    except (RuntimeError, OSError) as e:
        _fail(f"전사 실패: {e}")
    if not segments:
        _fail("전사 결과가 비어 있습니다 — 오디오에 음성이 없거나 형식이 잘못됐을 수 있습니다.")

    # ── 화자 분리 (선택) ───────────────────────────────────────────────────
    if do_diarize:
        from meetscribe.diarize import assign_speakers, diarize
        if progress:
            print("화자 분리 중...")
        try:
            turns = diarize(audio_path, hf_token=hf_token)
            segments = assign_speakers(segments, turns)
        except (RuntimeError, OSError) as e:
            _eprint(f"경고: 화자 분리 실패 — 화자 없이 진행합니다 ({e})")

    # ── 요약 (선택, 현재 비활성) ───────────────────────────────────────────
    summary = None
    if do_summarize:
        from meetscribe.summarize import summarize
        try:
            summary = summarize(segments, model=args.llm_model,
                                host=args.llm_host, progress=progress)
        except RuntimeError as e:
            _eprint(f"경고: 요약 생략 ({e})")

    # ── 렌더 + 저장 ────────────────────────────────────────────────────────
    meta = parse_meta(audio_path)
    md = render_markdown(segments, meta, summary=summary)

    try:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
    except OSError as e:
        _fail(f"저장 실패: {out_path} ({e})")
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
