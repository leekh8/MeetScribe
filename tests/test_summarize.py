"""summarize — 로컬 LLM 요약 (Ollama 백엔드)."""

import json

import pytest

from meetscribe import summarize as S


def _seg(text, start=0.0, speaker=None):
    seg = {"start": start, "end": start + 1, "text": text}
    if speaker:
        seg["speaker"] = speaker
    return seg


# ── 전사본 조립 ──────────────────────────────────────────────────────────────

def test_transcript_includes_speaker_when_present():
    text = S._segments_to_transcript([_seg("첫 마디", speaker="A"), _seg("둘째 마디")])
    assert text == "A: 첫 마디\n둘째 마디"


# ── 조각내기 ────────────────────────────────────────────────────────────────

def test_chunk_keeps_lines_intact():
    text = "\n".join(f"{i}번 줄입니다" for i in range(20))
    chunks = S.chunk_transcript(text, max_chars=40)
    assert len(chunks) > 1
    # 어느 조각에서도 줄이 잘리지 않아야 한다.
    assert set("\n".join(chunks).splitlines()) == set(text.splitlines())


def test_chunk_does_not_split_a_single_long_line():
    long_line = "가" * 500
    assert S.chunk_transcript(long_line, max_chars=100) == [long_line]


def test_chunk_drops_blank_result():
    assert S.chunk_transcript("") == []


# ── 모델 출력 파싱 ──────────────────────────────────────────────────────────

def test_parse_json_plain():
    assert S._parse_json('{"a": 1}') == {"a": 1}


def test_parse_json_strips_code_fence():
    # format=json을 줘도 코드펜스를 두르는 경우가 있다.
    assert S._parse_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_json_finds_object_in_prose():
    assert S._parse_json('정리했습니다.\n{"a": 1}\n이상입니다.') == {"a": 1}


def test_parse_json_returns_empty_on_garbage():
    assert S._parse_json("모르겠습니다") == {}
    assert S._parse_json("") == {}


def test_parse_json_rejects_non_object():
    # 배열을 돌려주면 dict 취급하지 않는다.
    assert S._parse_json("[1, 2, 3]") == {}


# ── 요약 본체 ───────────────────────────────────────────────────────────────

@pytest.fixture
def stub_backend(monkeypatch):
    """check_backend를 통과시키고 _chat 응답을 대신 준다."""
    calls = []

    def _install(responses):
        monkeypatch.setattr(S, "check_backend", lambda *a, **k: None)

        def fake_chat(prompt, system, model, host):
            calls.append(prompt)
            return responses[min(len(calls) - 1, len(responses) - 1)]

        monkeypatch.setattr(S, "_chat", fake_chat)
        return calls

    return _install


def test_summarize_empty_segments_skips_backend():
    assert S.summarize([]) == S.Summary()


def test_single_chunk_skips_the_merge_call(stub_backend):
    calls = stub_backend([{
        "topics": ["배포 일정"],
        "decisions": ["금요일에 배포한다"],
        "action_items": [{"owner": "홍길동", "task": "릴리스 노트", "due": "목요일"}],
    }])
    result = S.summarize([_seg("짧은 회의")], progress=False)
    assert len(calls) == 1                      # map만, reduce 없음
    assert result.decisions == ["금요일에 배포한다"]
    assert result.action_items[0]["owner"] == "홍길동"


def test_multiple_chunks_are_merged(stub_backend):
    calls = stub_backend([
        {"topics": ["A"], "decisions": ["결정 1"], "action_items": []},
        {"topics": ["B"], "decisions": ["결정 2"], "action_items": []},
        {"overview": "두 가지를 정했다", "decisions": ["결정 1", "결정 2"], "action_items": []},
    ])
    # 화자가 다르면 병합되지 않아 줄이 유지된다.
    segs = [_seg("가" * 15, 0, speaker="A"), _seg("나" * 15, 1, speaker="B")]
    result = S.summarize(segs, progress=False, chunk_chars=20)
    assert len(calls) == 3                       # map 2번 + reduce 1번
    assert result.overview == "두 가지를 정했다"
    assert result.decisions == ["결정 1", "결정 2"]


def test_failed_merge_keeps_chunk_results(stub_backend, monkeypatch):
    # 병합이 실패해도 조각 결과를 버리지 않는다. 중복이 남는 게 빈 요약보다 낫다.
    responses = [
        {"decisions": ["결정 1"], "action_items": [{"task": "일 1"}]},
        {"decisions": ["결정 2"], "action_items": []},
    ]
    calls = []

    monkeypatch.setattr(S, "check_backend", lambda *a, **k: None)

    def fake_chat(prompt, system, model, host):
        calls.append(prompt)
        return {} if "조각별 정리" in prompt else responses[min(len(calls) - 1, 1)]

    monkeypatch.setattr(S, "_chat", fake_chat)
    result = S.summarize([_seg("가" * 15, 0, speaker="A"), _seg("나" * 15, 1, speaker="B")],
                         progress=False, chunk_chars=20)
    assert result.decisions == ["결정 1", "결정 2"]
    assert result.action_items == [{"owner": "", "task": "일 1", "due": ""}]


def test_all_chunks_unparseable_raises(stub_backend):
    stub_backend([{}])
    with pytest.raises(RuntimeError, match="비어 있습니다"):
        S.summarize([_seg("무엇")], progress=False)


def test_malformed_lists_are_ignored(stub_backend):
    # 모델이 배열 대신 문자열을 주는 경우가 있다. 크래시하지 않아야 한다.
    stub_backend([{"topics": "주제", "decisions": None, "action_items": ["문자열"]}])
    result = S.summarize([_seg("무엇")], progress=False)
    assert result.decisions == []
    assert result.action_items == []


# ── 백엔드 점검 ─────────────────────────────────────────────────────────────

def test_check_backend_reports_missing_model(monkeypatch):
    class FakeResponse:
        def read(self):
            # Ollama /api/tags 실제 응답 형태.
            return json.dumps({"models": [{"name": "llama3:8b"}]}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(S.urllib.request, "urlopen", lambda *a, **k: FakeResponse())
    with pytest.raises(RuntimeError, match="ollama pull"):
        S.check_backend(model="gemma3:4b")


def test_check_backend_accepts_latest_suffix(monkeypatch):
    class FakeResponse:
        def read(self):
            return json.dumps({"models": [{"name": "gemma3:4b:latest"}]}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(S.urllib.request, "urlopen", lambda *a, **k: FakeResponse())
    S.check_backend(model="gemma3:4b")


def test_check_backend_explains_connection_failure(monkeypatch):
    def boom(*a, **k):
        raise S.urllib.error.URLError("연결 거부")

    monkeypatch.setattr(S.urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError, match="ollama serve"):
        S.check_backend()


def test_merge_joins_short_utterances_of_the_same_speaker():
    segs = [_seg("짧은 말", i, speaker="A") for i in range(5)]
    merged = S._segments_to_transcript(segs, merge_chars=200)
    assert merged == "A: 짧은 말 짧은 말 짧은 말 짧은 말 짧은 말"


def test_merge_breaks_when_speaker_changes():
    segs = [_seg("가", 0, speaker="A"), _seg("나", 1, speaker="B"), _seg("다", 2, speaker="A")]
    assert S._segments_to_transcript(segs, merge_chars=200).splitlines() == [
        "A: 가", "B: 나", "A: 다",
    ]


def test_merge_is_off_by_default():
    segs = [_seg("짧은 말", i, speaker="A") for i in range(3)]
    assert len(S._segments_to_transcript(segs).splitlines()) == 3


def test_merge_skips_blank_text():
    assert S._segments_to_transcript([_seg("  ", 0), _seg("내용", 1)], 200) == "내용"


def test_overview_falls_back_to_topics_when_merge_returns_none(monkeypatch):
    # 몇 분을 쓰고 빈 개요를 받지 않도록 조각의 주제로 메운다.
    monkeypatch.setattr(S, "check_backend", lambda *a, **k: None)
    responses = [
        {"topics": ["OpenVAS 패턴"], "decisions": [], "action_items": []},
        {"topics": ["Nuclei 템플릿"], "decisions": [], "action_items": []},
        {"overview": "", "decisions": [], "action_items": []},
    ]
    calls = []

    def fake_chat(prompt, system, model, host):
        calls.append(prompt)
        return responses[min(len(calls) - 1, 2)]

    monkeypatch.setattr(S, "_chat", fake_chat)
    result = S.summarize([_seg("가" * 15, 0, speaker="A"), _seg("나" * 15, 1, speaker="B")],
                         progress=False, chunk_chars=20)
    assert result.overview == "OpenVAS 패턴, Nuclei 템플릿"


# ── 지어낸 값 걸러내기 ──────────────────────────────────────────────────────

def test_placeholder_words_become_empty(stub_backend):
    # 모델이 "owner": "빈 문자열" 처럼 설명어를 값에 그대로 적는 경우가 있다.
    stub_backend([{
        "topics": ["주제"],
        "decisions": ["결정", "없음", "   "],
        "action_items": [{"owner": "빈 문자열", "task": "할 일", "due": "미정"}],
    }])
    result = S.summarize([_seg("무엇")], progress=False)
    assert result.decisions == ["결정"]
    assert result.action_items == [{"owner": "", "task": "할 일", "due": ""}]


def test_action_item_without_task_is_dropped(stub_backend):
    stub_backend([{
        "topics": [],
        "decisions": [],
        "action_items": [{"owner": "홍길동", "task": "", "due": "월요일"},
                         {"owner": "", "task": "실제 할 일", "due": ""}],
    }])
    result = S.summarize([_seg("무엇")], progress=False)
    assert result.action_items == [{"owner": "", "task": "실제 할 일", "due": ""}]


def test_action_items_are_normalised_to_three_fields(stub_backend):
    # 모델이 키를 빠뜨리거나 더 넣어도 렌더가 기대하는 세 칸으로 맞춘다.
    stub_backend([{"topics": [], "decisions": [],
                   "action_items": [{"task": "할 일", "priority": "high"}]}])
    result = S.summarize([_seg("무엇")], progress=False)
    assert result.action_items == [{"owner": "", "task": "할 일", "due": ""}]
