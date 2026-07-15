"""render — 세그먼트 → 마크다운 회의록."""

from pathlib import Path

from meetscribe.render import _cell, parse_meta, render_markdown
from meetscribe.summarize import Summary


def test_parse_meta_extracts_date():
    meta = parse_meta(Path("meeting_260327_160355.m4a"))
    assert meta["date"] == "2026-03-27"


def test_parse_meta_no_date_uses_stem_as_title():
    meta = parse_meta(Path("randomname.m4a"))
    assert meta["date"] == ""
    assert meta["title"] == "randomname"


def test_parse_meta_strips_recorder_suffix():
    assert parse_meta(Path("주간회의 Recording.m4a"))["title"] == "주간회의"


def test_cell_escapes_pipe_and_newline():
    assert _cell("a|b") == r"a\|b"
    assert _cell("a\nb") == "a b"
    assert _cell("  x  ") == "x"


def test_render_transcript_only():
    segs = [{"start": 5, "end": 10, "text": "안녕하세요"}]
    md = render_markdown(segs, {"title": "T", "date": "2026-01-01"})
    assert "# T" in md
    assert "**[00:05]** 안녕하세요" in md
    # 요약이 없으면 요약 섹션도 없어야 한다
    assert "## 요약" not in md


def test_render_with_speaker():
    segs = [{"start": 0, "end": 3, "text": "네", "speaker": "SPEAKER_00"}]
    md = render_markdown(segs, {"title": "T"})
    assert "**[00:00] SPEAKER_00:** 네" in md


def test_render_with_summary_sections():
    segs = [{"start": 0, "end": 1, "text": "x"}]
    summary = Summary(
        overview="개요 텍스트",
        decisions=["결정1"],
        action_items=[{"owner": "규주", "task": "할일", "due": "내일"}],
    )
    md = render_markdown(segs, {"title": "T"}, summary=summary)
    assert "## 요약" in md and "개요 텍스트" in md
    assert "## 결정사항" in md and "- 결정1" in md
    assert "## 액션아이템" in md and "| 규주 | 할일 | 내일 |" in md
    assert "## 전문" in md
