"""vocab — 교정 후보 발굴."""

import json

from meetscribe.vocab import (
    Candidate,
    find_candidates,
    load_document,
    merge_answers,
    render_questions,
    strip_josa,
    tokenize,
    translit_ratio,
)


def _doc(name, sentences, speakers=()):
    return {"name": name, "sentences": list(sentences), "speakers": set(speakers)}


# ── 토큰 ────────────────────────────────────────────────────────────────────

def test_strip_josa_merges_inflections():
    assert strip_josa("포티소어를") == "포티소어"
    assert strip_josa("포티소어가") == "포티소어"


def test_strip_josa_keeps_short_stem():
    # 조사를 떼면 1음절만 남는 경우는 원형을 지킨다.
    assert strip_josa("나를") == "나를"


def test_tokenize_splits_hangul_and_latin():
    tokens = tokenize("FortiSOAR 플레이북을 v7.2.1에서 확인")
    assert "FortiSOAR" in tokens
    assert "플레이북" in tokens
    assert "v7.2.1" in tokens


def test_tokenize_strips_plural_suffix():
    # '데이터'와 '데이터들이'가 한 토큰으로 모여야 빈도가 쪼개지지 않는다.
    assert "데이터" in tokenize("데이터들이 많다")


def test_translit_ratio_separates_loanwords():
    assert translit_ratio("스크립트") > translit_ratio("먹었다")


# ── 발굴 ────────────────────────────────────────────────────────────────────

def test_variant_group_catches_misheard_proper_noun():
    docs = [
        # 묶음의 각 표기가 최소 2회는 나와야 후보로 본다. 1회짜리는 단순 오탈자와 못 가른다.
        _doc("a", ["스프랑크가 탐지했다", "스프랑크 이벤트", "스프렁크도 마찬가지", "스프렁크 확인"]),
        _doc("b", ["오늘 날씨가 좋다"]),
    ]
    variants = {c.token: c for c in find_candidates(docs, {}) if c.kind == "variant"}
    assert "스프랑크" in variants
    assert "스프렁크" in variants["스프랑크"].siblings


def test_variant_ignores_verb_conjugation():
    # '확인해/확인하'처럼 마지막 음절이 어미면 고유명사 오인식이 아니다.
    docs = [_doc("a", ["확인해 주세요", "확인해 봤다", "확인하는 중", "확인하고 있다"])]
    tokens = {c.token for c in find_candidates(docs, {}) if c.kind == "variant"}
    assert "확인해" not in tokens


def test_variant_requires_same_first_syllable():
    # 첫 음절이 다르면 다른 단어로 들은 것이라 한 묶음으로 묻지 않는다.
    docs = [_doc("a", ["하고 있다 하고 있다", "가고 있다 가고 있다"])]
    assert not [c for c in find_candidates(docs, {}) if c.kind == "variant"]


def test_domain_skips_words_common_to_every_meeting():
    docs = [_doc(str(i), ["그래서 그거 이제 하나 " * 3]) for i in range(5)]
    assert not find_candidates(docs, {})


def test_known_terms_are_not_asked_again():
    docs = [_doc("a", ["워팔라이저 확인 " * 4]), _doc("b", ["다른 회의"])]
    assert any(c.token == "워팔라이저" for c in find_candidates(docs, {}))
    # 사전에 등재되면 다시 묻지 않는다.
    assert not any(
        c.token == "워팔라이저"
        for c in find_candidates(docs, {"워팔라이저": "Wappalyzer"})
    )


def test_correct_spelling_is_not_asked_either():
    # 사전의 값(정답 표기)도 이미 아는 말이므로 후보에서 빠진다.
    docs = [_doc("a", ["Wappalyzer 확인 " * 4]), _doc("b", ["다른 회의"])]
    assert not any(
        c.token == "Wappalyzer"
        for c in find_candidates(docs, {"워팔라이저": "Wappalyzer"})
    )


def test_speaker_names_are_excluded():
    docs = [_doc("a", ["김진성 님이 말했다 " * 4], speakers=["김진성"]), _doc("b", ["다른 회의"])]
    assert not any(c.token == "김진성" for c in find_candidates(docs, {}))


def test_latin_case_variants_are_grouped():
    docs = [
        _doc("a", ["fortisoar 설치", "FortiSOAR 설치", "FortiSOAR 확인"]),
        _doc("b", ["다른 회의"]),
    ]
    latin = {c.token: c for c in find_candidates(docs, {}) if c.kind == "latin"}
    assert "FortiSOAR" in latin
    assert "fortisoar" in latin["FortiSOAR"].siblings


def test_limit_caps_result_size():
    docs = [_doc("a", [f"용어{i} 스크립트 " * 3 for i in range(50)]), _doc("b", ["다른 회의"])]
    assert len(find_candidates(docs, {}, limit=5)) <= 5


# ── 입출력 ──────────────────────────────────────────────────────────────────

def test_load_document_parses_md_transcript(tmp_path):
    md = tmp_path / "회의.md"
    md.write_text(
        "---\nstatus: draft\n---\n\n"
        "**[00:01]** **이규해** 첫 문장입니다.\n"
        "**[00:05]** **박주현**: 둘째 문장입니다.\n"
        "본문 아닌 줄\n",
        encoding="utf-8",
    )
    doc = load_document(md)
    assert doc["sentences"] == ["첫 문장입니다.", "둘째 문장입니다."]
    assert doc["speakers"] == {"이규해", "박주현"}


def test_load_document_parses_json_transcript(tmp_path):
    js = tmp_path / "회의.json"
    js.write_text(
        json.dumps({"segments": [{"start": 0, "end": 1, "text": "안녕하세요"}]}),
        encoding="utf-8",
    )
    assert load_document(js)["sentences"] == ["안녕하세요"]


def test_render_questions_lists_every_candidate():
    cands = [
        Candidate("스프랑크", "variant", 4, 1, 9.0, ["스프랑크가 탐지"], ["스프렁크"]),
        Candidate("워팔라이저", "domain", 6, 1, 8.0, ["워팔라이저에서"]),
    ]
    md = render_questions(cands, [_doc("a", ["x"])])
    assert "스프랑크 / 스프렁크" in md
    assert "워팔라이저" in md
    assert md.count("정확한 표기:") == 2


def test_merge_answers_creates_and_updates(tmp_path):
    path = tmp_path / "corrections.local.json"
    assert merge_answers({"스프랑크": "Splunk"}, path) == (1, 0, 0)
    assert json.loads(path.read_text(encoding="utf-8")) == {"스프랑크": "Splunk"}

    # 같은 값은 갱신으로 세지 않고, 새 항목만 추가된다.
    assert merge_answers({"스프랑크": "Splunk", "소아": "SOAR"}, path) == (1, 0, 0)
    assert merge_answers({"스프랑크": "splunk"}, path) == (0, 1, 0)
    assert json.loads(path.read_text(encoding="utf-8"))["스프랑크"] == "splunk"


def test_merge_answers_keeps_existing_entries(tmp_path):
    path = tmp_path / "corrections.local.json"
    path.write_text(json.dumps({"기존": "Existing"}, ensure_ascii=False), encoding="utf-8")
    merge_answers({"신규": "New"}, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"기존": "Existing", "신규": "New"}


def test_merge_answers_skips_empty_and_identity(tmp_path):
    path = tmp_path / "corrections.local.json"
    assert merge_answers({"": "X", "Y": "", "같음": "같음"}, path) == (0, 0, 0)
    assert not path.exists()


def test_empty_answer_goes_to_the_ignore_list(tmp_path):
    # 오인식이 아니라는 답(인명 등)은 사전이 아니라 무시 목록으로 간다.
    dict_path = tmp_path / "corrections.local.json"
    ignore_path = tmp_path / "vocab_ignore.local.json"
    assert merge_answers({"에이미": "", "소아": "SOAR"}, dict_path, ignore_path) == (1, 0, 1)
    assert json.loads(ignore_path.read_text(encoding="utf-8")) == ["에이미"]
    assert "에이미" not in json.loads(dict_path.read_text(encoding="utf-8"))


def test_ignore_list_is_not_duplicated(tmp_path):
    dict_path = tmp_path / "corrections.local.json"
    ignore_path = tmp_path / "vocab_ignore.local.json"
    merge_answers({"에이미": ""}, dict_path, ignore_path)
    assert merge_answers({"에이미": ""}, dict_path, ignore_path) == (0, 0, 0)


def test_ignored_terms_are_not_asked_again():
    docs = [_doc("a", ["에이미 확인 " * 4]), _doc("b", ["다른 회의"])]
    assert any(c.token == "에이미" for c in find_candidates(docs, {}))
    assert not any(c.token == "에이미"
                   for c in find_candidates(docs, {}, ignored={"에이미"}))


def test_people_surface_forms_expands_aliases_and_parts():
    from meetscribe.vocab import people_surface_forms
    forms = people_surface_forms({"이규해": ["규해 주임", "규주", "에이미"]})
    # 통째 호칭과 그 조각이 모두 들어간다.
    assert {"이규해", "규해 주임", "규주", "에이미", "규해", "주임"} <= forms


def test_people_forms_keep_every_alias_out_of_candidates():
    from meetscribe.vocab import people_surface_forms
    docs = [
        _doc("a", ["규해 주임이 확인했다 " * 3, "규주가 답했다 " * 3, "주현 팀장도 봤다 " * 3]),
        _doc("b", ["다른 회의"]),
    ]
    forms = people_surface_forms({
        "이규해": ["규해 주임", "규주"],
        "박주현": ["주현 팀장"],
    })
    tokens = {c.token for c in find_candidates(docs, {}, ignored=forms)}
    assert not tokens & {"규해", "규주", "주현", "주임", "팀장"}


def test_people_forms_ignore_malformed_entries():
    from meetscribe.vocab import people_surface_forms
    assert people_surface_forms({"이름": [], "  ": ["  "]}) == {"이름"}
