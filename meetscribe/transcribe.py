"""faster-whisper 로컬 전사.

세그먼트 리스트(dict: start/end/text)를 반환한다. 화자·요약은 이후 단계에서 덧붙인다.
전사 결과에 도메인 용어 교정을 즉시 적용한다.
"""

from pathlib import Path

from .audio import format_ts, get_duration
from .corrections import Corrector

# VAD(음성 구간 감지) 기본값 — 조용한 발화도 놓치지 않도록 threshold를 낮춘다.
_VAD_PARAMS = dict(min_silence_duration_ms=300, speech_pad_ms=300, threshold=0.3)


def transcribe(
    audio_path: Path,
    model_size: str = "medium",
    language: str = "ko",
    corrector: Corrector | None = None,
    progress: bool = True,
) -> list[dict]:
    """단일 오디오 파일을 전사해 교정된 세그먼트 리스트를 돌려준다."""
    from faster_whisper import WhisperModel

    corrector = corrector if corrector is not None else Corrector()
    duration = get_duration(audio_path)

    if progress:
        print(f"모델 로딩: {model_size} (CPU, int8)")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def _run(vad: bool):
        segments_iter, _info = model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
            word_timestamps=True,
            vad_filter=vad,
            vad_parameters=_VAD_PARAMS if vad else None,
        )
        out = []
        for seg in segments_iter:
            text = corrector.apply(seg.text.strip())
            out.append({"start": seg.start, "end": seg.end, "text": text})
            if progress:
                pct = min(seg.end / duration * 100, 100) if duration > 0 else 0
                print(f"\r  [{pct:5.1f}%] {format_ts(seg.start)} {text[:60]}", end="")
        return out

    segments = _run(vad=True)
    # VAD가 전 구간을 침묵으로 오판하면 세그먼트 0개 → VAD 끄고 재시도.
    if not segments:
        if progress:
            print("\n  VAD로 감지 실패 — VAD 없이 재시도")
        segments = _run(vad=False)

    if progress:
        print(f"\n전사 완료: {len(segments)}개 세그먼트")
    return segments
