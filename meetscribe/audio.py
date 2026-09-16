"""오디오 유틸 — 길이 측정, 타임스탬프 포맷."""

import subprocess
from datetime import timedelta
from pathlib import Path


def get_duration(path: Path) -> float:
    """ffprobe로 오디오 길이(초). 알 수 없으면 0.0 반환(진행률 표시 생략용).

    과거엔 파일 크기로 '≈1MB/분' 추정했으나, 압축 코덱(m4a/mp3)에선 실제와 한 자릿수 이상
    어긋나 진행률이 크게 왜곡됐다. 부정확한 %보다 '알 수 없음(0.0)'이 정직하다.
    """
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return 0.0
        return float(result.stdout.strip())
    except (subprocess.SubprocessError, ValueError, OSError):
        return 0.0

def to_wav_16k_mono(src: Path, dst: Path) -> Path:
    """화자 분리 모델이 먹는 형식(16kHz 모노 16bit wav)으로 변환한다.

    ffmpeg 인자는 리스트로 넘긴다. 파일명에 공백이나 따옴표, 세미콜론이 들어간 녹음이
    흔한데 셸을 거치면 그게 명령으로 읽힌다. shell=False가 기본이라 그대로 두면 된다.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(src),
             "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
             "-loglevel", "error", str(dst)],
            capture_output=True, text=True, timeout=1800,
        )
    except FileNotFoundError:
        raise RuntimeError("ffmpeg을 찾을 수 없습니다. 화자 분리에는 ffmpeg이 필요합니다.")
    except (subprocess.SubprocessError, OSError) as e:
        raise RuntimeError(f"오디오 변환 실패: {e}")
    if result.returncode != 0 or not dst.exists():
        raise RuntimeError(f"오디오 변환 실패: {result.stderr.strip()[:200]}")
    return dst


def format_ts(seconds: float) -> str:
    """초 → MM:SS 또는 H:MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


def format_duration(seconds: float) -> str:
    """초 → H:MM:SS."""
    return str(timedelta(seconds=int(seconds)))
