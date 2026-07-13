"""세그먼트 → 마크다운 회의록 렌더링.

화자(speaker)와 요약(Summary)이 있으면 반영하고, 없으면 생략한다 —
Phase 1(전사만)부터 Phase 2·3(화자·요약)까지 같은 렌더러로 커버한다.
"""

import re
from datetime import date
from pathlib import Path

from .audio import format_ts
from .summarize import Summary


def parse_meta(audio_path: Path) -> dict:
    """파일명에서 제목·날짜 추출. 'XXX_YYMMDD_HHMMSS' 패턴을 인식한다."""
    name = audio_path.stem
    meta = {"title": name, "date": ""}
    m = re.search(r"(\d{6})_(\d{6})", name)
    if m:
        d = m.group(1)
        meta["date"] = f"20{d[:2]}-{d[2:4]}-{d[4:6]}"
    # 'Voice 260327_160355', 'Call recording' 같은 녹음앱 접미어 제거
    title = re.sub(r"_?(Voice|Call recording|Recording)_?\s*\d*_?\d*", "", name, flags=re.I)
    meta["title"] = title.strip("_ ") or name
    return meta


def render_markdown(segments: list[dict], meta: dict,
                    summary: Summary | None = None) -> str:
    """회의록 마크다운 문자열 생성."""
    lines = [
        "---",
        "tags: [회의록]",
        f"date: {meta.get('date') or date.today().isoformat()}",
        f"title: \"{meta.get('title', '')}\"",
        "---",
        "",
        f"# {meta.get('title', '회의록')}",
        "",
    ]

    if summary is not None:
        lines += ["## 요약", "", summary.overview, ""]
        if summary.decisions:
            lines += ["## 결정사항", ""]
            lines += [f"- {d}" for d in summary.decisions]
            lines.append("")
        if summary.action_items:
            lines += ["## 액션아이템", "", "| 담당 | 내용 | 기한 |", "|---|---|---|"]
            for a in summary.action_items:
                lines.append(f"| {a.get('owner','')} | {a.get('task','')} | {a.get('due','')} |")
            lines.append("")
        lines += ["## 전문", ""]

    for seg in segments:
        ts = format_ts(seg["start"])
        speaker = seg.get("speaker")
        if speaker:
            lines.append(f"**[{ts}] {speaker}:** {seg['text']}")
        else:
            lines.append(f"**[{ts}]** {seg['text']}")
        lines.append("")

    return "\n".join(lines)
