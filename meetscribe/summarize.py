"""LLM 요약·액션아이템 추출 (Phase 3).

전사 전문을 받아 요약 / 결정사항 / 액션아이템을 뽑는다.
백엔드는 교체 가능(Claude API 기본, 향후 로컬 LLM). 자격증명이 없으면 명확히 중단한다.
"""

from dataclasses import dataclass, field


@dataclass
class Summary:
    overview: str = ""
    decisions: list[str] = field(default_factory=list)
    action_items: list[dict] = field(default_factory=list)  # {owner, task, due}


def _segments_to_transcript(segments: list[dict]) -> str:
    lines = []
    for seg in segments:
        speaker = seg.get("speaker")
        prefix = f"{speaker}: " if speaker else ""
        lines.append(f"{prefix}{seg['text']}")
    return "\n".join(lines)


def summarize(segments: list[dict], api_key: str | None = None,
              model: str = "claude-opus-4-8") -> Summary:
    """전사 → 요약. (Phase 3 구현 예정)

    Claude API로 요약/결정/액션아이템을 구조화 추출한다. 프롬프트·스키마는 Phase 3에서 확정.
    """
    try:
        import anthropic  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "LLM 요약에는 anthropic 패키지가 필요합니다.\n"
            "  1) requirements.txt에서 anthropic 주석 해제 후 pip install\n"
            "  2) ANTHROPIC_API_KEY 환경변수 또는 --api-key 로 키 전달"
        )
    raise NotImplementedError("summarize()는 Phase 3에서 구현 예정 — 전사 구조는 이미 준비됨")
