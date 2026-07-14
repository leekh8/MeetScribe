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

def format_ts(seconds: float) -> str:
    """초 → MM:SS 또는 H:MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


def format_duration(seconds: float) -> str:
    """초 → H:MM:SS."""
    return str(timedelta(seconds=int(seconds)))
