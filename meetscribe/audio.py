"""오디오 유틸 — 길이 측정, 타임스탬프 포맷."""

import subprocess
from datetime import timedelta
from pathlib import Path


def get_duration(path: Path) -> float:
    """ffprobe로 오디오 길이(초). 실패 시 파일 크기로 대략 추정."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=10,
        )
        return float(result.stdout.strip())
    except (subprocess.SubprocessError, ValueError, OSError):
        return path.stat().st_size / 1024 / 1024 * 60  # ≈ 1MB/분

def format_ts(seconds: float) -> str:
    """초 → MM:SS 또는 H:MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


def format_duration(seconds: float) -> str:
    """초 → H:MM:SS."""
    return str(timedelta(seconds=int(seconds)))
