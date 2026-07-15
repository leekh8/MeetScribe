"""audio — 타임스탬프·길이 포맷 (순수 함수만; ffprobe 호출은 제외)."""

from meetscribe.audio import format_duration, format_ts


def test_format_ts_under_hour():
    assert format_ts(0) == "00:00"
    assert format_ts(65) == "01:05"


def test_format_ts_over_hour():
    assert format_ts(3665) == "1:01:05"


def test_format_ts_truncates_fraction():
    assert format_ts(59.9) == "00:59"


def test_format_duration():
    assert format_duration(3665) == "1:01:05"
