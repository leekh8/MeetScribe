"""MeetScribe CLI — 오디오 → 마크다운 회의록.

    python cli.py 회의녹음.m4a
    python cli.py 회의녹음.m4a --model large-v3 -o out/
    python cli.py 회의녹음.m4a --diarize --summarize
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

# Windows 콘솔(cp949)에서도 한글이 깨지지 않도록 stdout/stderr를 UTF-8로 재설정.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from meetscribe import __version__
from meetscribe.config import DEFAULT_LANGUAGE, DEFAULT_MODEL, DEFAULT_OUTPUT_DIR
from meetscribe.render import (
    DEFAULT_SESSION,
    SESSION_TAGS,
    parse_meta,
    render_markdown,
    session_tag,
)
from meetscribe.summarize import DEFAULT_HOST as LLM_HOST
from meetscribe.summarize import DEFAULT_MODEL as LLM_MODEL
from meetscribe.transcribe import transcribe

# faster-whisper/ffmpeg가 디코드할 수 있는 대표 확장자.
SUPPORTED_EXTS = {".m4a", ".wav", ".mp3", ".flac", ".ogg", ".aac", ".mp4", ".webm", ".opus"}


def _dump_segments(segments: list, path: Path) -> None:
    """전사 결과를 그대로 떨궈 둔다. 전사는 가장 비싼 단계라 한 번 하면 다시 안 하게."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(segments, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError as e:
        _eprint(f"경고: 전사본 저장 실패 ({e})")


def _load_segments(path: Path) -> list:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        _fail(f"전사본을 읽을 수 없음: {path} ({e})")
    if not isinstance(data, list) or not data:
        _fail(f"전사본이 비었거나 형식이 다릅니다: {path}")
    return data


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

    from meetscribe.config import load_corrections, load_ignored, load_people
    from meetscribe.vocab import (
        find_candidates,
        load_corpus,
        people_surface_forms,
        render_questions,
    )

    missing = [p for p in paths if not p.exists()]
    if missing:
        _fail(f"경로 없음: {', '.join(str(p) for p in missing)}")

    docs = load_corpus(paths)
    if not docs:
        _fail("읽을 전사본이 없습니다 (md 또는 json 필요).")

    skip = load_ignored() | people_surface_forms(load_people())
    candidates = find_candidates(docs, load_corrections(), ignored=skip, limit=limit)
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

    from meetscribe.config import LOCAL_DICT_PATH, LOCAL_IGNORE_PATH
    from meetscribe.vocab import merge_answers

    if not answers_path.exists():
        _fail(f"파일 없음: {answers_path}")
    try:
        answers = _json.loads(answers_path.read_text(encoding="utf-8"))
    except (_json.JSONDecodeError, OSError) as e:
        _fail(f"답변 파일을 읽을 수 없습니다: {e}")
    if not isinstance(answers, dict):
        _fail('답변 파일은 {"오인식": "정답"} 형태의 객체여야 합니다.')

    added, updated, ignored = merge_answers(answers, LOCAL_DICT_PATH, LOCAL_IGNORE_PATH)
    print(f"사전 반영: 추가 {added}건 / 갱신 {updated}건 → {LOCAL_DICT_PATH}")
    if ignored:
        print(f"무시 목록: {ignored}건 추가 → {LOCAL_IGNORE_PATH} (값이 빈 항목)")


def main():
    parser = argparse.ArgumentParser(description="MeetScribe — 로컬 회의록 자동화")
    parser.add_argument("audio", nargs="?", help="오디오 파일 (m4a/wav/mp3 등)")
    parser.add_argument("--suggest-terms", nargs="+", metavar="PATH", default=None,
                        help="전사본(md/json)에서 교정 후보를 뽑아 질문지 생성")
    parser.add_argument("--apply-terms", metavar="JSON", type=Path, default=None,
                        help='답변 JSON({"오인식":"정답"})을 병합. '
                             '값이 비면 오인식이 아닌 것으로 보고 무시 목록에 넣는다')
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
    parser.add_argument("--diarize-backend", choices=["onnx", "pyannote"], default="onnx",
                        help="화자 분리 백엔드 (기본 onnx: 모델 74MB, 토큰 불필요). "
                             "pyannote는 torch 2.5GB와 HuggingFace 토큰이 필요하다")
    parser.add_argument("--attendees", default=None, metavar="이름,이름",
                        help="참석자 (쉼표 구분). 담당자 대조 기준에 더해져 액션아이템의 "
                             "담당 칸이 채워진다")
    parser.add_argument("--speakers", type=int, default=None, metavar="N",
                        help="참석자 수. 미지정 시 --attendees 개수를 쓴다. "
                             "자동 추정은 신뢰할 수 없으므로 아는 값이면 지정할 것")
    parser.add_argument("--type", dest="session_type", default=DEFAULT_SESSION,
                        metavar="종류",
                        help="세션 종류: " + " / ".join(SESSION_TAGS)
                             + " 또는 임의 문자열(그대로 태그가 된다)")
    parser.add_argument("--from-transcript", type=Path, default=None, metavar="JSON",
                        help="저장해 둔 전사본 json으로 시작. 전사를 건너뛰므로 "
                             "요약이나 양식만 다시 돌릴 때 쓴다")
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
    segments_path = args.out_dir / f"{audio_path.stem}.segments.json"
    # --from-transcript는 이미 만든 회의록을 다시 만드는 것이 목적이라 --force를 요구하지 않는다.
    # 양식이나 요약만 바꿔 돌려보는 경로에 매번 --force를 붙이게 하면 재시도가 비싸진다.
    if out_path.exists() and not args.force and not args.dry_run and not args.from_transcript:
        _fail(f"출력 파일이 이미 있음: {out_path} (덮어쓰려면 --force)")
    if args.from_transcript and not args.from_transcript.exists():
        _fail(f"전사본 없음: {args.from_transcript}")

    # 참석자는 회의 시작 전에 사람이 아는 값이다. 자동 추정에 맡기지 않고 받는다.
    attendees = [n.strip() for n in (args.attendees or "").split(",") if n.strip()]
    num_speakers = args.speakers or (len(attendees) or None)

    # ── 선택 기능 사전 점검 (비싼 전사 전에 실패/경고) ──────────────────────
    # 전사는 수 분~수십 분 걸리므로, 쓸 수 없는 옵션은 여기서 걸러 낭비를 막는다.
    do_diarize = args.diarize
    do_summarize = args.summarize
    if do_diarize and args.diarize_backend == "onnx":
        from meetscribe.diarize import EMB_MODEL, SEG_MODEL
        if not _module_available("sherpa_onnx"):
            _eprint("경고: sherpa-onnx 미설치로 화자 분리를 건너뜁니다 "
                    "(pip install sherpa-onnx numpy).")
            do_diarize = False
        elif not (SEG_MODEL.exists() and EMB_MODEL.exists()):
            _eprint(f"경고: 화자 분리 모델이 없어 건너뜁니다 ({SEG_MODEL.parent.parent}). "
                    "README의 모델 내려받기 절차를 보세요.")
            do_diarize = False
    elif do_diarize and not _module_available("pyannote.audio"):
        _eprint("경고: pyannote.audio 미설치로 화자 분리를 건너뜁니다 "
                "(requirements.txt 주석 해제 후 설치).")
        do_diarize = False
    if do_diarize and not num_speakers:
        # 자동 추정은 정답 3명짜리 녹음에서 28명까지밖에 못 좁혔다. 결과를 못 믿는다는 것을
        # 미리 말한다. 조용히 이상한 결과를 내는 것보다 낫다.
        _eprint("경고: 참석자 수를 모르면 화자 수 추정이 크게 빗나갑니다. "
                "--speakers N 또는 --attendees 로 알려주세요.")
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
        print(f"[dry-run] 종류 : {args.session_type} (태그 {session_tag(args.session_type)})"
              f" / 참석 {len(attendees)}명"
              + (f" ({', '.join(attendees)})" if attendees else "")
              + (f" / 화자 수 {num_speakers}" if num_speakers else ""))
        if args.from_transcript:
            print(f"[dry-run] 전사 : 건너뜀, {args.from_transcript} 재사용")
        return

    progress = not args.quiet
    meta = parse_meta(audio_path)

    # 단계마다 같은 경로에 다시 쓴다.
    #
    # 전사는 수십 분, 요약은 30~40분, 화자 분리는 그 이상 걸린다. 맨 끝에서 한 번만 쓰면
    # 그 사이 어디서 죽든(Ctrl-C 포함) 앞 단계 결과가 통째로 사라진다. 완성본이 아니어도
    # 그 시점까지 확보한 것은 디스크에 있어야 한다.
    def save(segs, summary=None) -> None:
        md = render_markdown(segs, meta, summary=summary,
                             attendees=attendees, session_type=args.session_type)
        try:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            out_path.write_text(md, encoding="utf-8")
        except OSError as e:
            _fail(f"저장 실패: {out_path} ({e})")

    # ── 전사 (또는 저장된 전사본 재사용) ───────────────────────────────────
    if args.from_transcript:
        segments = _load_segments(args.from_transcript)
        if progress:
            print(f"전사본 재사용: {args.from_transcript} ({len(segments)}개 구간)")
    else:
        try:
            segments = transcribe(audio_path, model_size=args.model,
                                  language=args.language, progress=progress)
        except (RuntimeError, OSError) as e:
            _fail(f"전사 실패: {e}")
        if not segments:
            _fail("전사 결과가 비어 있습니다. 오디오에 음성이 없거나 형식이 잘못됐을 수 있습니다.")
        _dump_segments(segments, segments_path)
        if progress:
            print(f"전사본 저장: {segments_path} (--from-transcript 로 재사용 가능)")

    save(segments)
    if progress:
        print(f"1차 저장: {out_path} (전사까지)")

    # ── 화자 분리 (선택) ───────────────────────────────────────────────────
    if do_diarize:
        from meetscribe.diarize import assign_speakers, diarize, diarize_onnx
        if progress:
            print(f"화자 분리 중... ({args.diarize_backend}"
                  + (f", {num_speakers}명 고정)" if num_speakers else ", 화자 수 추정)"))
        try:
            if args.diarize_backend == "onnx":
                turns = diarize_onnx(audio_path, num_speakers=num_speakers)
            else:
                turns = diarize(audio_path, hf_token=hf_token, num_speakers=num_speakers)
            segments = assign_speakers(segments, turns)
            save(segments)
            if progress:
                found = len({t["speaker"] for t in turns})
                print(f"화자 {found}명 / 구간 {len(turns)}개 -> {out_path}")
        except (RuntimeError, OSError) as e:
            _eprint(f"경고: 화자 분리 실패로 화자 없이 진행합니다 ({e})")

    # ── 요약 (선택) ────────────────────────────────────────────────────────
    summary = None
    if do_summarize:
        from meetscribe.summarize import summarize
        try:
            summary = summarize(segments, model=args.llm_model,
                                host=args.llm_host, progress=progress,
                                attendees=attendees)
        except RuntimeError as e:
            _eprint(f"경고: 요약 생략 ({e})")

    save(segments, summary=summary)
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
