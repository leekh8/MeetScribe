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
