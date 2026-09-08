"""Corrector — 도메인 용어 후처리 교정."""

from meetscribe.corrections import Corrector


def test_basic_substitution():
    c = Corrector({"시엠": "SIEM"})
    assert c.apply("시엠 구축 회의") == "SIEM 구축 회의"


def test_case_insensitive():
    c = Corrector({"docker": "Docker"})
    assert c.apply("DOCKER 이미지") == "Docker 이미지"


def test_longest_match_first():
    # '오픈' 단독 규칙이 '오픈 바스'를 먼저 깨뜨리면 안 된다 — 긴 표현 우선.
    c = Corrector({"오픈": "X", "오픈 바스": "OpenVAS"})
    assert c.apply("오픈 바스 스캔") == "OpenVAS 스캔"


def test_replacement_with_backslash_is_literal():
    # 치환값에 역슬래시/그룹참조 표기가 있어도 정규식 백참조로 해석되지 않아야 한다.
    c = Corrector({"foo": r"a\1b"})
    assert c.apply("foo") == r"a\1b"


def test_len_reports_rule_count():
    assert len(Corrector({"a": "b", "c": "d"})) == 2


def test_no_match_leaves_text_unchanged():
    c = Corrector({"시엠": "SIEM"})
    assert c.apply("관련 없는 문장") == "관련 없는 문장"


def test_ascii_key_requires_word_boundary():
    # UR -> URL 규칙이 URL을 URLL로 만들면 안 된다.
    c = Corrector({"UR": "URL"})
    assert c.apply("참조 UR을 본문에") == "참조 URL을 본문에"
    assert c.apply("참조 URL을 본문에") == "참조 URL을 본문에"


def test_ascii_key_does_not_match_inside_english_word():
    # IGNORECASE라 during 안의 ur까지 잡히면 dURLing이 된다.
    c = Corrector({"UR": "URL"})
    assert c.apply("during the meeting") == "during the meeting"


def test_ascii_key_still_matches_next_to_hangul_particle():
    c = Corrector({"FDM": "FDN"})
    assert c.apply("FDM 통신과 FDM이") == "FDN 통신과 FDN이"


def test_hangul_key_keeps_substring_behaviour():
    # 한글은 조사가 붙어 오므로 경계를 걸지 않는다.
    c = Corrector({"소아": "SOAR"})
    assert c.apply("소아에서 처리하는 걸") == "SOAR에서 처리하는 걸"


def test_hangul_key_does_not_match_inside_a_word():
    # '바스' -> OpenVAS 규칙이 '자바스크립트'를 깨뜨리면 안 된다.
    c = Corrector({"바스": "OpenVAS"})
    assert c.apply("자바스크립트 확인") == "자바스크립트 확인"
    assert c.apply("바스는 아직 안 뺐고") == "OpenVAS는 아직 안 뺐고"


def test_hangul_key_matches_at_word_start_after_space():
    c = Corrector({"바스": "OpenVAS"})
    assert c.apply("뉴클레이랑 바스") == "뉴클레이랑 OpenVAS"


def test_longer_rule_wins_over_prefixed_form():
    # '오픈바스'는 앞이 한글이라 '바스' 규칙에 안 걸리므로 자기 규칙이 필요하다.
    c = Corrector({"바스": "OpenVAS", "오픈바스": "OpenVAS"})
    assert c.apply("오픈바스 패턴 정리") == "OpenVAS 패턴 정리"
