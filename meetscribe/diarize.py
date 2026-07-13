"""화자 분리 (Phase 2 — pyannote.audio).

전사 세그먼트에 "누가 말했나"를 태깅한다. pyannote 모델은 대용량이고 HuggingFace
토큰 + 모델 사용 동의가 필요하므로, 의존성이 없으면 명확한 안내와 함께 중단한다.

인터페이스:
    turns = diarize(audio_path)                      # [{start, end, speaker}]
    segments = assign_speakers(segments, turns)      # 각 세그먼트에 speaker 부여
"""

from pathlib import Path


def diarize(audio_path: Path, hf_token: str | None = None) -> list[dict]:
    """화자 구간 리스트 [{start, end, speaker}] 반환. (Phase 2 구현 예정)"""
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        raise RuntimeError(
            "화자 분리에는 pyannote.audio가 필요합니다.\n"
            "  1) requirements.txt에서 pyannote 주석 해제 후 pip install\n"
            "  2) HuggingFace 토큰 발급 + pyannote/speaker-diarization 모델 사용 동의\n"
            "  3) --hf-token 으로 토큰 전달 (또는 HF_TOKEN 환경변수)"
        )

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
    )
    annotation = pipeline(str(audio_path))
    turns = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        turns.append({"start": turn.start, "end": turn.end, "speaker": speaker})
    return turns


def assign_speakers(segments: list[dict], turns: list[dict]) -> list[dict]:
    """각 전사 세그먼트에 가장 많이 겹치는 화자를 부여한다."""
    for seg in segments:
        best_speaker, best_overlap = None, 0.0
        for turn in turns:
            overlap = min(seg["end"], turn["end"]) - max(seg["start"], turn["start"])
            if overlap > best_overlap:
                best_overlap, best_speaker = overlap, turn["speaker"]
        seg["speaker"] = best_speaker
    return segments
