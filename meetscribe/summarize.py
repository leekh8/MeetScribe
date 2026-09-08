"""LLM 요약, 액션아이템 추출 (Phase 3).

로컬 Ollama로 돌린다. 이 프로젝트의 전제는 "녹음이 기기 밖으로 나가지 않는다"이므로
요약만 외부 API로 보내면 그 전제가 무너진다. 품질이 아니라 그것이 로컬을 쓰는 이유다.

한 시간짜리 회의는 한 번에 넣지 않는다. 조각별로 뽑고(map) 한 번 더 합친다(reduce).
GPU 없이 도는 것을 전제로 하므로 호출 횟수를 줄이는 쪽으로 조각을 크게 잡는다.

의존성을 늘리지 않으려고 HTTP는 표준 라이브러리로 직접 부른다.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field

DEFAULT_MODEL = "gemma3:4b"
DEFAULT_HOST = "http://localhost:11434"

# CPU 추론은 느리다. 조각 하나에 수 분이 걸릴 수 있어 넉넉히 잡는다.
CALL_TIMEOUT = 900
CHUNK_CHARS = 3500
# 짧은 발화를 이 길이까지 이어 붙인다. 전사본은 발화당 7~15자로 잘게 끊겨 있어
# 그대로 넣으면 모델이 문맥을 못 잡고 한 줄에 꽂힌다.
MERGE_CHARS = 200


@dataclass
class Summary:
    overview: str = ""
    decisions: list[str] = field(default_factory=list)
    action_items: list[dict] = field(default_factory=list)  # {owner, task, due}


def _segments_to_transcript(segments: list[dict], merge_chars: int = 0) -> str:
    """세그먼트를 전사본 텍스트로. merge_chars를 주면 짧은 발화를 이어 붙인다.

    전사본은 발화당 평균 7~15자로 잘게 끊겨 있다. 그대로 넣으면 모델이 문맥 없는
    수백 개의 조각을 보게 되어, 실제 논의를 놓치고 한 줄에 꽂혀 엉뚱한 것을 뽑는다.
    화자가 바뀌면 끊고, 같은 화자 안에서만 길이로 묶는다.
    """
    lines: list[list[str]] = []
    for seg in segments:
        speaker = seg.get("speaker") or ""
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        if (merge_chars and lines and lines[-1][0] == speaker
                and len(lines[-1][1]) < merge_chars):
            lines[-1][1] = f"{lines[-1][1]} {text}"
        else:
            prefix = f"{speaker}: " if speaker else ""
            lines.append([speaker, f"{prefix}{text}"])
    return "\n".join(line for _, line in lines)


# ── 백엔드 ──────────────────────────────────────────────────────────────────

def check_backend(host: str = DEFAULT_HOST, model: str = DEFAULT_MODEL) -> None:
    """모델이 준비됐는지 미리 본다. 전사는 수십 분이 걸리므로 그 전에 실패시킨다."""
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=10) as resp:
            tags = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Ollama에 연결할 수 없습니다 ({host}): {e.reason}\n"
            "  ollama serve 로 데몬을 띄우거나 Ollama 앱이 실행 중인지 확인하세요."
        ) from e

    names = {m.get("name", "") for m in tags.get("models", [])}
    if model not in names and f"{model}:latest" not in names:
        raise RuntimeError(
            f"모델 '{model}'이 없습니다. 받은 모델: {', '.join(sorted(names)) or '없음'}\n"
            f"  ollama pull {model}"
        )


def _chat(prompt: str, system: str, model: str, host: str) -> dict:
    """Ollama에 한 번 물어 JSON 객체를 받는다."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "format": "json",
        # 요약은 창작이 아니다. 온도를 0으로 두어 같은 입력에 같은 결과가 나오게 한다.
        "options": {"temperature": 0},
    }
    req = urllib.request.Request(
        f"{host}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=CALL_TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama 호출 실패: {e}") from e
    return _parse_json(body.get("message", {}).get("content", ""))


def _parse_json(text: str) -> dict:
    """모델 출력에서 JSON 객체를 꺼낸다.

    format=json을 줘도 코드펜스를 두르거나 앞뒤에 말을 붙이는 경우가 있다.
    파싱이 끝내 안 되면 빈 dict를 돌려준다. 한 조각 실패로 전체를 버리지 않는다.
    """
    text = (text or "").strip()
    if not text:
        return {}
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return {}
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {}
    return parsed if isinstance(parsed, dict) else {}


# ── 조각내기 ────────────────────────────────────────────────────────────────

def chunk_transcript(text: str, max_chars: int = CHUNK_CHARS) -> list[str]:
    """줄 경계를 지키며 자른다. 문장이 잘리면 모델이 앞뒤를 지어낸다."""
    chunks, current, size = [], [], 0
    for line in text.splitlines():
        # 한 줄이 통째로 한도를 넘으면 그 줄만 따로 둔다. 억지로 쪼개지 않는다.
        if size and size + len(line) + 1 > max_chars:
            chunks.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return [c for c in chunks if c.strip()]


# ── 프롬프트 ────────────────────────────────────────────────────────────────

_SYSTEM = (
    "너는 한국어 회의록 정리 담당이다. 전사본은 음성인식 결과라 오탈자가 있다. "
    "추측해서 내용을 지어내지 말고, 전사본에 실제로 있는 말만 쓴다. "
    "반드시 JSON 객체 하나만 출력한다."
)

_MAP_PROMPT = """다음은 회의 전사본의 일부다. 이 부분에서만 뽑아라.

{{
  "topics": ["다룬 주제를 짧은 구로"],
  "decisions": ["확정된 결정만. 논의 중인 것은 넣지 않는다"],
  "action_items": [{{"owner": "", "task": "할 일", "due": ""}}]
}}

규칙:
- 해당 항목이 없으면 빈 배열로 둔다. 잡담이나 진행 대화만 있는 부분에는 결정도 액션도 없다.
  억지로 채우지 마라. 빈 배열이 정답인 경우가 많다.
- owner는 전사본에 이름이 실제로 나온 경우에만 적는다. "개발자", "담당자", "팀 리더" 같은
  직책을 지어내지 마라. 모르면 "" 로 둔다.
- due는 전사본에 날짜나 요일이 실제로 나온 경우에만 적는다. 날짜를 추측하지 마라.
  모르면 "" 로 둔다.
- "빈 문자열", "없음", "미정" 같은 말을 값으로 쓰지 마라. 비어 있으면 "" 다.
- 한 문장에만 나오고 앞뒤와 이어지지 않는 말은 뽑지 않는다. 전사 오류일 가능성이 크다.

전사본:
{chunk}"""

_REDUCE_PROMPT = """아래는 한 회의를 여러 조각으로 나눠 정리한 결과다. 하나로 합쳐라.

{{
  "overview": "회의 전체를 3~5문장으로",
  "decisions": ["중복을 제거한 결정사항"],
  "action_items": [{{"owner": "", "task": "", "due": ""}}]
}}

규칙:
- 같은 내용이 여러 조각에 있으면 하나로 합친다.
- 조각에 없는 내용을 새로 만들지 않는다. 특히 owner와 due는 조각에 있는 값만 옮긴다.
  비어 있던 것을 채우지 마라.
- "빈 문자열", "없음", "미정" 같은 말을 값으로 쓰지 마라. 비어 있으면 "" 다.

조각별 정리:
{parts}"""


# ── 본체 ────────────────────────────────────────────────────────────────────

def _as_list(value) -> list:
    return value if isinstance(value, list) else []


# 모델이 값 자리에 설명어를 그대로 적는 경우가 있다("owner": "빈 문자열").
# 프롬프트로만 막으면 새는 날이 있어 코드에서 한 번 더 거른다.
_PLACEHOLDER = {"", "-", "빈 문자열", "빈문자열", "없음", "미정", "해당 없음", "해당없음",
                "unknown", "n/a", "na", "null", "none", "tbd"}


def _clean(value) -> str:
    text = str(value if value is not None else "").strip()
    return "" if text.lower() in _PLACEHOLDER else text


def _clean_actions(items) -> list[dict]:
    """액션아이템을 owner/task/due 세 칸으로 정규화한다. 할 일이 없으면 버린다."""
    cleaned = []
    for item in _as_list(items):
        if not isinstance(item, dict):
            continue
        task = _clean(item.get("task"))
        if not task:
            continue
        cleaned.append({
            "owner": _clean(item.get("owner")),
            "task": task,
            "due": _clean(item.get("due")),
        })
    return cleaned


def _clean_decisions(items) -> list[str]:
    return [d for d in (_clean(x) for x in _as_list(items)) if d]


def summarize(segments: list[dict], *, model: str = DEFAULT_MODEL,
              host: str = DEFAULT_HOST, progress: bool = True,
              chunk_chars: int = CHUNK_CHARS) -> Summary:
    """전사 세그먼트 → 요약, 결정사항, 액션아이템."""
    if not segments:
        return Summary()

    check_backend(host, model)
    chunks = chunk_transcript(_segments_to_transcript(segments, MERGE_CHARS), chunk_chars)

    parts = []
    for i, chunk in enumerate(chunks, 1):
        if progress:
            print(f"요약 {i}/{len(chunks)} 조각...")
        result = _chat(_MAP_PROMPT.format(chunk=chunk), _SYSTEM, model, host)
        if result:
            parts.append(result)

    if not parts:
        raise RuntimeError("요약 결과가 비어 있습니다 — 모델 응답을 해석하지 못했습니다.")

    # 조각이 하나뿐이면 합칠 것이 없다. 호출 한 번을 아낀다.
    if len(parts) == 1:
        only = parts[0]
        return Summary(
            overview=", ".join(dict.fromkeys(
                t for t in (_clean(x) for x in _as_list(only.get("topics"))) if t)),
            decisions=_clean_decisions(only.get("decisions")),
            action_items=_clean_actions(only.get("action_items")),
        )

    if progress:
        print(f"조각 {len(parts)}개 병합...")
    merged = _chat(
        _REDUCE_PROMPT.format(parts=json.dumps(parts, ensure_ascii=False, indent=1)),
        _SYSTEM, model, host,
    )

    # 병합이 실패해도 조각 결과는 살린다. 중복이 남더라도 빈 요약보다 낫다.
    if not merged:
        return Summary(
            overview="",
            decisions=[d for p in parts for d in _clean_decisions(p.get("decisions"))],
            action_items=[a for p in parts for a in _clean_actions(p.get("action_items"))],
        )

    overview = str(merged.get("overview", "")).strip()
    if not overview:
        # 병합이 개요를 비워 오는 경우가 있다. 수 분을 쓰고 빈 요약을 받지 않게 주제라도 남긴다.
        topics = [t for p in parts for t in
                  (_clean(x) for x in _as_list(p.get("topics"))) if t]
        overview = ", ".join(dict.fromkeys(topics))
    return Summary(
        overview=overview,
        decisions=_clean_decisions(merged.get("decisions")),
        action_items=_clean_actions(merged.get("action_items")),
    )
