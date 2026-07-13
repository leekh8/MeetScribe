"""MeetScribe CLI — 오디오 → 마크다운 회의록.

    python cli.py 회의녹음.m4a
    python cli.py 회의녹음.m4a --model large-v3 -o out/
    python cli.py 회의녹음.m4a --diarize --summarize
"""

import argparse
import io
import os
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from meetscribe.config import DEFAULT_MODEL, DEFAULT_OUTPUT_DIR
from meetscribe.render import parse_meta, render_markdown
from meetscribe.transcribe import transcribe


def main():
    parser = argparse.ArgumentParser(description="MeetScribe — 로컬 회의록 자동화")
    parser.add_argument("audio", help="오디오 파일 (m4a/wav/mp3)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        choices=["tiny", "base", "small", "medium", "large-v3"],
                        help=f"Whisper 모델 (기본 {DEFAULT_MODEL})")
    parser.add_argument("-o", "--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="출력 디렉토리")
    parser.add_argument("--diarize", action="store_true", help="화자 분리 (Phase 2)")
    parser.add_argument("--summarize", action="store_true", help="LLM 요약 (Phase 3)")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                        help="화자 분리용 HuggingFace 토큰")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"ERROR: 파일 없음: {audio_path}")
        sys.exit(1)

    segments = transcribe(audio_path, model_size=args.model)

    if args.diarize:
        from meetscribe.diarize import assign_speakers, diarize
        print("화자 분리 중...")
        turns = diarize(audio_path, hf_token=args.hf_token)
        segments = assign_speakers(segments, turns)

    summary = None
    if args.summarize:
        from meetscribe.summarize import summarize
        print("요약 생성 중...")
        summary = summarize(segments)

    meta = parse_meta(audio_path)
    md = render_markdown(segments, meta, summary=summary)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{audio_path.stem}.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
