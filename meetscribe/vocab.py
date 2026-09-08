"""교정 후보 발굴.

전사본을 사람이 처음부터 읽는 대신, 도메인 용어 오인식으로 보이는 토큰만 뽑아
"이 단어 맞습니까"로 물어볼 목록을 만든다. 사람의 일은 전문 검토가 아니라 질문 N개 답변이 된다.

신호 세 가지를 쓴다.
  1. domain  회의별로 편중된 토큰(tf-idf). 일상어는 모든 회의에 고르게 나와 자동으로 걸러진다
  2. variant 한 글자만 다른 한글 토큰 묶음. 고유명사를 매번 다르게 알아들은 흔적이다
  3. latin   대소문자만 다른 영문 표기 흔들림

셋 다 모델이 필요 없다. 로컬 LLM 백엔드는 이 후보 위에 얹는 선택 단계다.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# 한글 토큰과 영문 토큰을 따로 잡는다. 영문은 버전 표기(v7.2.1)와 하이픈 이름을 살린다.
_HANGUL = re.compile(r"[가-힣]{2,}")
_LATIN = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[.-][A-Za-z0-9]+)*")

# 전사 MD 한 줄: **[MM:SS]** **화자** 텍스트  (화자 뒤 콜론은 있을 수도 없을 수도)
_MD_LINE = re.compile(r"^\*\*\[[\d:]+\]\*\*\s*(?:\*\*(?P<speaker>[^*]+)\*\*:?)?\s*(?P<text>.+)$")

# 조사를 떼야 "포티소어를"과 "포티소어가"가 한 토큰으로 모인다.
# 긴 것부터 검사한다. 어간이 2자 미만으로 줄어드는 절단은 하지 않는다.
_JOSA = sorted(
    ["으로써", "으로서", "에서는", "에게서", "이라고", "라고는",
     "으로", "에서", "에게", "한테", "부터", "까지", "보다", "처럼", "이라", "라고",
     "이나", "이든", "이며", "이고", "인데", "이야", "이지",
     "은", "는", "이", "가", "을", "를", "와", "과", "의", "도", "만", "에", "로",
     "요", "죠", "님", "씨", "들"],
    key=len, reverse=True,
)

# 도메인 후보로 볼 이유가 없는 표면형. 회의 진행어라 tf-idf만으로는 남는 것들이다.
_STOP = {
    "그래서", "그러면", "그러니까", "그니까", "아니라", "이제", "약간", "일단", "그거",
    "저거", "이거", "뭐지", "뭐야", "맞아요", "그렇죠", "그쵸", "네네", "아니요",
    "지금", "다음", "오늘", "내일", "어제", "이번", "저번", "여기", "거기", "저기",
    "하나", "가지고", "해가지고", "그런데", "근데", "왜냐면", "말씀", "생각",
}

# 마지막 음절이 어미면 용언 활용형이다. 고유명사가 아니므로 변형 묶음에서 뺀다.
# 이 필터가 없으면 "하고/가고/갖고/같고"가 한 묶음이 되어 탐지기 전체가 무너진다.
_ENDING = set("고는서어을에지다게면요죠대도야네랑며니까라러려한할함했돼됐된될줘줄준봐본볼온올"
              "데거걸건예시를은겠잖")

# 바르게 전사된 흔한 말. 오인식이 아니므로 물어볼 이유가 없다.
# tf-idf만으로는 회의 주제에 따라 편중돼 상위로 올라온다.
_COMMON = {
    # 일상어
    "다시", "같이", "너무", "사실", "경우", "때문", "자체", "따로", "정도", "대신",
    "부분", "전체", "이상", "이하", "이후", "이전", "처음", "마지막", "여러", "각각",
    "조금", "많이", "제일", "가장", "실제", "결국", "물론", "혹시", "아마", "약간",
    "한번", "두번", "이유", "방법", "문제", "내용", "상황", "관련", "필요", "가능",
    "얘기", "이야기", "질문", "답변", "설명", "정리", "준비", "진행", "시작", "종료",
    "회의", "미팅", "오전", "오후", "이번주", "다음주", "지난주", "월요일", "화요일",
    # 바르게 전사되는 흔한 IT 한국어
    "서버", "파일", "정보", "확인", "처리", "관리", "시간", "내부", "외부", "설정",
    "화면", "메뉴", "버튼", "계정", "권한", "접속", "연동", "저장", "삭제", "수정",
    "등록", "조회", "검색", "결과", "목록", "항목", "기능", "구성", "환경", "버전",
    "장비", "고객", "담당", "요청", "제품", "지원", "테스트", "점검", "조치", "이슈",
}

# 활용형끼리 갈린 묶음을 걸러내는 어말. "확인해/확인하", "해드리/해드린/해드릴"처럼
# 마지막 음절에서만 갈리고 그 음절이 어미면 고유명사 오인식이 아니다.
_VERB_TAIL = set("하해한할함했리린릴려러가갈감와워줘줄준어아오지")

# 영어를 한글로 옮길 때 유난히 자주 나오는 음절. 고유명사 후보를 가려내는 약한 신호다.
_TRANSLIT = set("스트드크프브즈츠쓰티디리러라로루시샤셔쇼슈이에엘엠엔엑씨비제젤케코컴콘"
                "포파퍼펌푸퓨후휴워웨위웨뉴네노누마머모무바베보부밴번벤빌솔센션션스")


@dataclass
class Candidate:
    """물어볼 후보 하나."""

    token: str
    kind: str                 # domain | variant | latin
    count: int
    docs: int
    score: float
    contexts: list[str] = field(default_factory=list)
    siblings: list[str] = field(default_factory=list)   # variant/latin 묶음의 다른 표기

    def to_dict(self) -> dict:
        return {
            "token": self.token, "kind": self.kind, "count": self.count,
            "docs": self.docs, "score": round(self.score, 3),
            "siblings": self.siblings, "contexts": self.contexts,
        }


# ── 입력 ────────────────────────────────────────────────────────────────────

def load_document(path: Path) -> dict:
    """전사 MD 또는 JSON 한 건을 {name, sentences, speakers}로 읽는다."""
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        segments = raw.get("segments", raw) if isinstance(raw, dict) else raw
        sentences = [s.get("text", "").strip() for s in segments if isinstance(s, dict)]
        speakers: set[str] = set()
    else:
        sentences, speakers = [], set()
        for line in path.read_text(encoding="utf-8").splitlines():
            m = _MD_LINE.match(line.strip())
            if not m:
                continue
            sentences.append(m.group("text").strip())
            if m.group("speaker"):
                speakers.add(m.group("speaker").strip())
    return {"name": path.stem, "sentences": [s for s in sentences if s], "speakers": speakers}


def load_corpus(paths: list[Path]) -> list[dict]:
    """파일과 디렉토리를 섞어 받아 문서 목록으로 만든다.

    같은 회의의 md와 json이 함께 있으면 md를 택한다(화자명을 갖고 있어 인명 제외에 쓰인다).
    """
    files: list[Path] = []
    for p in paths:
        files.extend(sorted(p.rglob("*.md")) + sorted(p.rglob("*.json")) if p.is_dir() else [p])

    chosen: dict[str, Path] = {}
    for f in files:
        if f.suffix.lower() not in (".md", ".json"):
            continue
        prev = chosen.get(f.stem)
        if prev is None or (prev.suffix.lower() == ".json" and f.suffix.lower() == ".md"):
            chosen[f.stem] = f

    docs = []
    for f in sorted(chosen.values()):
        try:
            doc = load_document(f)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if doc["sentences"]:
            docs.append(doc)
    return docs


# ── 토큰 ────────────────────────────────────────────────────────────────────

def strip_josa(token: str) -> str:
    """조사를 하나만 뗀다. 어간이 2자 미만이 되면 원형을 유지한다."""
    for josa in _JOSA:
        if token.endswith(josa) and len(token) - len(josa) >= 2:
            return token[: -len(josa)]
    return token


def tokenize(text: str) -> list[str]:
    """한글은 조사와 복수 접미사를 떼고, 영문은 원형을 유지한 채 토큰을 뽑는다."""
    tokens = [strip_josa(strip_josa(t)) for t in _HANGUL.findall(text)]
    tokens.extend(_LATIN.findall(text))
    return [t for t in tokens if len(t) >= 2]


def _hangul_edit1(a: str, b: str) -> bool:
    """같은 고유명사를 다르게 알아들은 관계인가.

    단순 편집거리 1로는 안 된다. 한국어는 2음절 어절이 촘촘해 "하고"와 "가고"까지 묶인다.
    그래서 첫 음절 일치를 요구한다. 오인식은 뒤쪽 음절에서 갈리지, 첫 음절부터
    갈리면 애초에 다른 단어로 들은 것이라 한 묶음으로 물어볼 이유가 없다.
    """
    if not a or not b or a[0] != b[0]:
        return False
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    short, long = (a, b) if len(a) < len(b) else (b, a)
    # 첫 음절은 이미 같으므로 1번 위치부터 검사한다.
    return any(long[:i] + long[i + 1:] == short for i in range(1, len(long)))


def _is_inflected(token: str) -> bool:
    """용언 활용형으로 보이는가. 마지막 음절이 어미면 고유명사 후보에서 뺀다."""
    return bool(token) and token[-1] in _ENDING


def translit_ratio(token: str) -> float:
    """외래어 음차에 흔한 음절의 비율. 0에 가까우면 순우리말 쪽이다."""
    if not token:
        return 0.0
    return sum(ch in _TRANSLIT for ch in token) / len(token)


# ── 발굴 ────────────────────────────────────────────────────────────────────

def people_surface_forms(people: dict) -> set[str]:
    """호칭 전체와 그것을 쪼갠 조각까지 모은다.

    "길동 주임"은 띄어쓰기 때문에 "길동"와 "주임" 두 토큰으로 잡힌다.
    통째로만 빼면 조각이 그대로 후보에 남으므로 조각도 함께 넣는다.
    """
    forms: set[str] = set()
    for canonical, aliases in people.items():
        for name in [canonical, *aliases]:
            name = (name or "").strip()
            if not name:
                continue
            forms.add(name)
            forms.update(tokenize(name))
    return forms


def _known_surface(corrections: dict) -> set[str]:
    """이미 사전이 다루는 표기. 키와 값 양쪽 모두 다시 묻지 않는다."""
    known = set()
    for wrong, right in corrections.items():
        known.add(wrong.replace(" ", "").lower())
        known.add(right.replace(" ", "").lower())
    return known


def find_candidates(docs: list[dict], corrections: dict | None = None, *,
                    ignored: set | None = None,
                    min_count: int = 3, max_docs_ratio: float = 0.6,
                    limit: int = 60) -> list[Candidate]:
    """세 신호를 합쳐 물어볼 후보를 점수순으로 돌려준다."""
    corrections = corrections or {}
    known = _known_surface(corrections)
    # 화자 목록에 없는 인명은 계속 후보로 올라온다. 한 번 아니라고 하면 다시 묻지 않는다.
    people = {s for d in docs for s in d["speakers"]} | (ignored or set())

    n_docs = max(len(docs), 1)
    tf: Counter[str] = Counter()
    df: Counter[str] = Counter()
    context: dict[str, list[str]] = defaultdict(list)

    for doc in docs:
        seen_here: set[str] = set()
        for sentence in doc["sentences"]:
            for token in tokenize(sentence):
                tf[token] += 1
                seen_here.add(token)
                if len(context[token]) < 3 and len(sentence) <= 120:
                    context[token].append(sentence)
        for token in seen_here:
            df[token] += 1

    def eligible(token: str) -> bool:
        if token in _STOP or token in _COMMON or token in people:
            return False
        if token.replace(" ", "").lower() in known:
            return False
        # 사람 이름이 조사와 붙어 잘린 형태도 제외한다.
        return not any(token.startswith(p[:2]) and len(p) >= 2 and token in p for p in people)

    candidates: dict[str, Candidate] = {}

    def offer(token: str, kind: str, score: float, siblings: list[str] | None = None) -> None:
        prev = candidates.get(token)
        if prev is not None and prev.score >= score:
            return
        candidates[token] = Candidate(
            token=token, kind=kind, count=tf[token], docs=df[token], score=score,
            contexts=context.get(token, [])[:3], siblings=sorted(siblings or []),
        )

    # 1) domain — 회의별 편중. 모든 회의에 나오는 일상어는 idf가 0이라 자동 탈락한다.
    #    음차 비율을 곱해, 같은 빈도라면 외래어처럼 생긴 쪽을 먼저 묻는다.
    doc_cap = max(1, int(n_docs * max_docs_ratio))
    for token, count in tf.items():
        if count < min_count or df[token] > doc_cap or not eligible(token):
            continue
        if _HANGUL.fullmatch(token) and _is_inflected(token):
            continue
        idf = math.log(n_docs / df[token]) + 1.0
        boost = 1.0 + 2.0 * translit_ratio(token) if _HANGUL.fullmatch(token) else 2.0
        offer(token, "domain", count * idf * boost)

    # 2) variant — 첫 음절이 같고 한 음절만 다른 한글 토큰 묶음.
    #    고유명사를 매번 다르게 알아들은 흔적이다. 조건을 좁게 잡는다.
    #      - 3음절 이상: 2음절은 순우리말 밀도가 너무 높아 노이즈만 나온다
    #      - 활용형 제외: 어미로 끝나면 용언이다
    #      - 소수 회의 편중: 모든 회의에 나오면 일상어다
    #      - 묶음 4개 이하: 그 이상 붙으면 고유명사가 아니라 흔한 어절 뭉치다
    hangul = sorted(
        (t for t in tf
         if _HANGUL.fullmatch(t) and len(t) >= 3 and tf[t] >= 2
         and df[t] <= doc_cap and not _is_inflected(t) and eligible(t)),
        key=lambda t: (-tf[t], t),
    )
    grouped: set[str] = set()
    for i, base in enumerate(hangul):
        if base in grouped:
            continue
        siblings = [o for o in hangul[i + 1:] if o not in grouped and _hangul_edit1(base, o)]
        if not siblings or len(siblings) > 3:
            continue
        group = [base, *siblings]
        # 마지막 음절에서만 갈리는데 그 음절이 어미면 활용형 묶음이다.
        tail_only = all(len(m) == len(base) and m[:-1] == base[:-1] for m in siblings)
        if tail_only and any(m[-1] in _VERB_TAIL for m in group):
            grouped.update(group)
            continue
        grouped.update(group)
        total = tf[base] + sum(tf[s] for s in siblings)
        offer(base, "variant", total * (1 + len(siblings)) * 2.0, siblings)

    # 3) latin — 대소문자만 다른 영문 표기 흔들림.
    by_lower: dict[str, set[str]] = defaultdict(set)
    for token in tf:
        if _LATIN.fullmatch(token):
            by_lower[token.lower()].add(token)
    for forms in by_lower.values():
        if len(forms) < 2:
            continue
        ordered = sorted(forms, key=lambda t: (-tf[t], t))
        head = ordered[0]
        if not eligible(head):
            continue
        offer(head, "latin", sum(tf[f] for f in forms) * 2.0, ordered[1:])

    ranked = sorted(candidates.values(), key=lambda c: (-c.score, c.token))
    return ranked[:limit]


# ── 출력 ────────────────────────────────────────────────────────────────────

_KIND_LABEL = {
    "domain": "이 회의에만 몰려 나옴",
    "variant": "같은 말을 다르게 알아들은 흔적",
    "latin": "영문 표기 흔들림",
}


def render_questions(candidates: list[Candidate], docs: list[dict]) -> str:
    """사람에게 그대로 보여 줄 질문지 마크다운."""
    lines = [
        "# 용어 확인 요청",
        "",
        f"전사본 {len(docs)}건에서 뽑은 교정 후보 {len(candidates)}개.",
        "맞으면 넘어가고, 틀린 것만 정확한 표기를 적어 주면 사전에 반영된다.",
        "",
    ]
    for kind in ("variant", "latin", "domain"):
        group = [c for c in candidates if c.kind == kind]
        if not group:
            continue
        lines += [f"## {_KIND_LABEL[kind]} ({len(group)}개)", ""]
        for i, c in enumerate(group, 1):
            forms = " / ".join([c.token, *c.siblings]) if c.siblings else c.token
            lines.append(f"{i}. **{forms}** (총 {c.count}회, 회의 {c.docs}건)")
            for ctx in c.contexts:
                lines.append(f"   - {ctx}")
            lines.append("   - 정확한 표기: ")
            lines.append("")
    return "\n".join(lines)


def merge_answers(answers: dict, path: Path,
                  ignore_path: Path | None = None) -> tuple[int, int, int]:
    """답변을 병합한다. (추가, 갱신, 무시등록) 건수를 돌려준다.

    값이 비어 있으면 "오인식이 아니다"라는 뜻으로 보고 무시 목록에 넣는다.
    인명처럼 교정 대상이 아닌 말을 매번 다시 묻지 않기 위한 경로다.

    기존 파일을 통째로 갈아엎지 않는다. 이미 있는 항목은 값이 달라질 때만 덮어쓴다.
    """
    current: dict = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                current = loaded
        except (json.JSONDecodeError, OSError):
            current = {}

    ignored: set[str] = set()
    if ignore_path is not None and ignore_path.exists():
        try:
            loaded = json.loads(ignore_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                ignored = {str(x) for x in loaded}
        except (json.JSONDecodeError, OSError):
            ignored = set()

    added = updated = skipped = 0
    for wrong, right in answers.items():
        wrong = (wrong or "").strip()
        right = (right or "").strip() if right is not None else ""
        if not wrong:
            continue
        if not right:
            # 오인식이 아니라는 답. 사전이 아니라 무시 목록으로 보낸다.
            if ignore_path is not None and wrong not in ignored:
                ignored.add(wrong)
                skipped += 1
            continue
        if wrong == right:
            continue
        if wrong not in current:
            added += 1
        elif current[wrong] != right:
            updated += 1
        else:
            continue
        current[wrong] = right

    if added or updated:
        path.write_text(
            json.dumps(dict(sorted(current.items())), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if skipped and ignore_path is not None:
        ignore_path.write_text(
            json.dumps(sorted(ignored), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return added, updated, skipped
