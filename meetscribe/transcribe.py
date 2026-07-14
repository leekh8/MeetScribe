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
    duration = get_duration(audio_path)  # 0이면 길이 불명(ffprobe 실패) → % 대신 타임스탬프 표시

    if progress:
        print(f"모델 로딩: {model_size} (CPU, int8)")
    try:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
    except Exception as e:  # 모델 다운로드 실패·네트워크·디스크 등
        raise RuntimeError(
            f"Whisper 모델 '{model_size}' 로드 실패 ({e}). "
            "최초 실행은 모델 다운로드가 필요하니 네트워크를 확인하세요."
        ) from e

    def _run(vad: bool):
        segments_iter, _info = model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
            word_timestamps=False,  # start/end/text만 사용 — 단어 단위 타임스탬프는 불필요한 오버헤드
            vad_filter=vad,
            vad_parameters=_VAD_PARAMS if vad else None,
        )
        out = []
        for seg in segments_iter:
            text = corrector.apply(seg.text.strip())
            out.append({"start": seg.start, "end": seg.end, "text": text})
            if progress:
                if duration > 0:
                    pct = min(seg.end / duration * 100, 100)
                    print(f"\r  [{pct:5.1f}%] {format_ts(seg.start)} {text[:60]}", end="")
                else:
                    print(f"\r  [{format_ts(seg.start)}] {text[:60]}", end="")
        return out

    try:
        segments = _run(vad=True)
        # VAD가 전 구간을 침묵으로 오판하면 세그먼트 0개 → VAD 끄고 재시도.
        if not segments:
            if progress:
                print("\n  VAD로 감지 실패 — VAD 없이 재시도")
            segments = _run(vad=False)
    except Exception as e:  # 디코드 단계 오류(ffmpeg 부재·손상 파일 등)
        raise RuntimeError(
            f"오디오 전사 실패 ({e}). ffmpeg 설치 여부와 오디오 파일 상태를 확인하세요."
        ) from e

    if progress:
        print(f"\n전사 완료: {len(segments)}개 세그먼트")
    return segments
