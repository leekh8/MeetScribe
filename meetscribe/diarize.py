"""화자 분리.

전사 세그먼트에 "누가 말했나"를 태깅한다. 백엔드가 둘이고 반환 형식이 같아서
하류(assign_speakers 이후)는 어느 쪽을 쓰든 바뀌지 않는다.

    diarize_onnx  sherpa-onnx. 모델 74MB, 토큰 불필요. 기본값이고 실측을 이쪽으로 했다.
    diarize       pyannote.audio. torch 약 2.5GB + HuggingFace 토큰과 모델 사용 동의.

어느 쪽이든 참석자 수를 주는 것이 정확도에 가장 크게 기여한다. 자동 추정은 폰 원거리
녹음에서 무너진다.

인터페이스:
    turns = diarize_onnx(audio_path, num_speakers=3)  # [{start, end, speaker}]
    segments = assign_speakers(segments, turns)       # 각 세그먼트에 speaker 부여
"""

import tempfile
import wave
from pathlib import Path

from .audio import to_wav_16k_mono
from .config import ROOT

# sherpa-onnx 경로. 모델은 용량이 커서 레포에 넣지 않는다(README에 받는 법).
SEG_MODEL = ROOT / "models" / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
# 임베딩은 ERes2Net을 기본으로 둔다. CAM++는 발화량이 적은 화자를 통째로 흡수했고
# (실측: 전체 발화의 15%를 차지한 화자가 1.5%로 뭉개졌다) ERes2Net으로 29%까지 복구됐다.
# min_duration_on을 흔들어도 효과가 없었으므로 임베딩 모델 자체가 갈림길이다.
EMB_MODEL = ROOT / "models" / "eres2net.onnx"


def _read_wav(path: Path):
    import numpy as np

    with wave.open(str(path)) as w:
        data = w.readframes(w.getnframes())
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def diarize_onnx(audio_path: Path, num_speakers: int | None = None,
                 seg_model: Path | None = None,
                 emb_model: Path | None = None) -> list[dict]:
    """sherpa-onnx로 화자 구간 [{start, end, speaker}]를 낸다.

    pyannote 경로보다 이쪽을 기본으로 두는 이유는 셋이다. 모델이 74MB로 torch 경로의
    1/34이고, HuggingFace 토큰과 모델 사용 동의 게이트가 없고, 실측을 이쪽으로 했다.

    num_speakers를 주면 군집 수를 그 값으로 고정한다. 안 주면 임계값으로 추정하는데
    폰 원거리 녹음에서는 이게 무너진다(정답 3명에 임계 0.5에서 64명, 0.8에서도 28명).
    참석자 수는 회의를 연 사람이 아는 값이므로 추정하지 말고 받는 편이 낫다.
    """
    try:
        import sherpa_onnx
    except ImportError:
        raise RuntimeError(
            "화자 분리에는 sherpa-onnx가 필요합니다.\n"
            "  pip install sherpa-onnx numpy"
        )

    seg = Path(seg_model) if seg_model else SEG_MODEL
    emb = Path(emb_model) if emb_model else EMB_MODEL
    for label, p in (("분할 모델", seg), ("임베딩 모델", emb)):
        if not p.exists():
            raise RuntimeError(f"{label}이 없습니다: {p}\n  README의 모델 내려받기 절차를 따르세요.")

    # 군집 수를 안 주면 임계값 추정으로 떨어진다. 결과를 못 믿는다는 것을 알리고 진행한다.
    clustering = (sherpa_onnx.FastClusteringConfig(num_clusters=num_speakers)
                  if num_speakers and num_speakers > 0
                  else sherpa_onnx.FastClusteringConfig(threshold=0.5))

    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(seg))),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(emb)),
        clustering=clustering,
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not cfg.validate():
        raise RuntimeError("화자 분리 설정 검증 실패. 모델 경로를 확인하세요.")

    with tempfile.TemporaryDirectory(prefix="meetscribe_") as tmp:
        wav = to_wav_16k_mono(audio_path, Path(tmp) / "audio.wav")
        samples = _read_wav(wav)
        result = sherpa_onnx.OfflineSpeakerDiarization(cfg).process(samples)

    return [{"start": s.start, "end": s.end, "speaker": f"SPEAKER_{s.speaker:02d}"}
            for s in result.sort_by_start_time()]


def diarize(audio_path: Path, hf_token: str | None = None,
            num_speakers: int | None = None) -> list[dict]:
    """화자 구간 리스트 [{start, end, speaker}] 반환.

    num_speakers를 주면 자동 추정을 건너뛴다. 폰으로 원거리 녹음한 회의에서는 자동
    추정이 무너져 정답 3명짜리 녹음에 수십 명이 나온다. 참석자 수는 회의를 연 사람이
    이미 아는 값이므로 추정하지 말고 받는 편이 낫다.
    """
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
    # 토큰 누락·무효거나 모델 사용 동의 전이면 pyannote는 예외 대신 None을 돌려준다.
    if pipeline is None:
        raise RuntimeError(
            "화자 분리 파이프라인 로드 실패 — HuggingFace 토큰이 유효한지, "
            "pyannote/speaker-diarization-3.1 모델 사용 동의를 마쳤는지 확인하세요."
        )
    kwargs = {"num_speakers": num_speakers} if num_speakers else {}
    annotation = pipeline(str(audio_path), **kwargs)
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
