"""세그먼트 → 마크다운 회의록 렌더링.

화자(speaker)와 요약(Summary)이 있으면 반영하고, 없으면 생략한다 —
Phase 1(전사만)부터 Phase 2·3(화자·요약)까지 같은 렌더러로 커버한다.
"""

import re
from datetime import date
from pathlib import Path

from .audio import format_ts
from .summarize import Summary


def _cell(value) -> str:
    """마크다운 표 셀 이스케이프 — 파이프·개행이 표 구조를 깨지 않게."""
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


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


# 세션 종류별 태그. 회의와 교육은 읽는 목적이 달라서 같은 태그로 묶으면 나중에 못 고른다.
# 여기 없는 값을 주면 그 문자열을 그대로 태그로 쓴다(양식을 코드가 가두지 않게).
SESSION_TAGS = {
    "meeting": "회의록",
    "training": "교육",
    "interview": "면담",
    "review": "검토",
}
DEFAULT_SESSION = "meeting"


def session_tag(session_type: str | None) -> str:
    key = (session_type or DEFAULT_SESSION).strip()
    return SESSION_TAGS.get(key, key or SESSION_TAGS[DEFAULT_SESSION])


def render_markdown(segments: list[dict], meta: dict,
                    summary: Summary | None = None,
                    attendees: list[str] | None = None,
                    session_type: str | None = None) -> str:
    """회의록 마크다운 문자열 생성."""
    names = [n.strip() for n in (attendees or []) if n and n.strip()]
    lines = [
        "---",
        f"tags: [{session_tag(session_type)}]",
        f"date: {meta.get('date') or date.today().isoformat()}",
        f"title: \"{meta.get('title', '')}\"",
    ]
    if names:
        lines.append("attendees: [" + ", ".join(names) + "]")
    lines += [
        "---",
        "",
        f"# {meta.get('title', '회의록')}",
        "",
    ]
    if names:
        lines += [f"참석 ({len(names)}명): " + ", ".join(names), ""]

    if summary is not None:
        lines += ["## 요약", "", summary.overview, ""]
        if summary.decisions:
            lines += ["## 결정사항", ""]
            lines += [f"- {d}" for d in summary.decisions]
            lines.append("")
        if summary.action_items:
            lines += ["## 액션아이템", "", "| 담당 | 내용 | 기한 |", "|---|---|---|"]
            for a in summary.action_items:
                lines.append(
                    f"| {_cell(a.get('owner',''))} | {_cell(a.get('task',''))} "
                    f"| {_cell(a.get('due',''))} |"
                )
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
