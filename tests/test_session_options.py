"""세션 옵션(참석자, 종류)과 전사본 재사용 검증.

지키려는 것 두 가지.
  1. 참석자를 받으면 액션아이템의 담당 칸이 더는 구조적으로 비지 않는다.
  2. 전사본을 파일로 떨궈 두면 전사를 다시 하지 않고 재개할 수 있다.
"""

import json

from meetscribe.render import DEFAULT_SESSION, render_markdown, session_tag
from meetscribe.summarize import Summary, _clean_actions, _grounding_source


class TestSessionTag:
    def test_알려진_종류는_한국어_태그로_바뀐다(self):
        assert session_tag("training") == "교육"
        assert session_tag("meeting") == "회의록"

    def test_미지정이면_기본값(self):
        assert session_tag(None) == session_tag(DEFAULT_SESSION) == "회의록"

    def test_모르는_값은_그대로_태그가_된다(self):
        # 양식을 코드가 가두지 않는다. 새 종류를 쓰려고 코드를 고칠 일이 없어야 한다.
        assert session_tag("워크숍") == "워크숍"

    def test_빈_문자열은_기본값으로_떨어진다(self):
        assert session_tag("   ") == "회의록"


class TestRenderAttendees:
    seg = [{"start": 0.0, "end": 1.0, "text": "시작하겠습니다"}]

    def test_참석자를_주면_프론트매터와_본문에_들어간다(self):
        md = render_markdown(self.seg, {"title": "주간 점검"},
                             attendees=["홍길동", "김영희"], session_type="meeting")
        assert "attendees: [홍길동, 김영희]" in md
        assert "참석 (2명): 홍길동, 김영희" in md

    def test_참석자가_없으면_해당_줄이_아예_없다(self):
        md = render_markdown(self.seg, {"title": "주간 점검"})
        assert "attendees:" not in md
        assert "참석 (" not in md

    def test_빈_이름은_걸러진다(self):
        md = render_markdown(self.seg, {"title": "t"}, attendees=["홍길동", "", "  "])
        assert "참석 (1명): 홍길동" in md

    def test_종류가_태그에_반영된다(self):
        md = render_markdown(self.seg, {"title": "t"}, session_type="training")
        assert "tags: [교육]" in md


class TestGroundingWithAttendees:
    """담당자 칸이 왜 비었는지, 참석자를 주면 왜 살아나는지."""

    transcript = "다음 주까지 방화벽 정책을 정리하기로 했습니다"

    def test_전사본에_없는_이름은_참석자를_안_주면_버려진다(self):
        items = [{"owner": "홍길동", "task": "방화벽 정책 정리", "due": "다음 주"}]
        got = _clean_actions(items, self.transcript)
        assert got[0]["owner"] == "", "화자 분리를 안 켜면 전사본에 이름이 없어 전부 빈다"
        assert got[0]["due"] == "다음 주", "원문에 있는 기한은 살아남는다"

    def test_참석자로_준_이름은_살아남는다(self):
        source = _grounding_source(self.transcript, ["홍길동", "김영희"])
        items = [{"owner": "홍길동", "task": "방화벽 정책 정리", "due": "다음 주"}]
        assert _clean_actions(items, source)[0]["owner"] == "홍길동"

    def test_참석자에도_전사본에도_없는_이름은_여전히_버린다(self):
        # 환각 차단선은 그대로여야 한다. 참석자 명단은 대조 기준을 넓힐 뿐이다.
        source = _grounding_source(self.transcript, ["홍길동"])
        items = [{"owner": "이철수", "task": "방화벽 정책 정리", "due": ""}]
        assert _clean_actions(items, source)[0]["owner"] == ""

    def test_회의에_없던_사람이_언급되면_담당자로_남는다(self):
        """참석자 명단이 담당자를 참석자로 한정하지는 않는다.

        자리에 없는 사람에게 일을 넘기는 경우가 흔하다("그건 김 과장이 처리하기로").
        그 이름은 전사본에 실제로 나오므로 명단에 없어도 통과해야 한다. 명단은
        전사본을 대체하는 것이 아니라 거기에 더해지는 것이다.
        """
        transcript = "그건 김 과장이 처리하기로 했습니다"
        source = _grounding_source(transcript, ["홍길동", "김영희"])  # 김 과장은 비참석
        items = [{"owner": "김 과장", "task": "처리", "due": ""}]
        assert _clean_actions(items, source)[0]["owner"] == "김 과장"

    def test_참석자가_비면_전사본_그대로다(self):
        assert _grounding_source(self.transcript, []) == self.transcript

    def test_빈_이름은_대조_기준에_안_들어간다(self):
        assert _grounding_source(self.transcript, ["  ", ""]) == self.transcript


class TestTranscriptRoundtrip:
    def test_떨군_전사본을_그대로_다시_읽는다(self, tmp_path):
        import cli

        segs = [{"start": 0.0, "end": 1.5, "text": "안녕하세요"},
                {"start": 1.5, "end": 3.0, "text": "시작하죠", "speaker": "SPEAKER_00"}]
        p = tmp_path / "sub" / "a.segments.json"
        cli._dump_segments(segs, p)

        assert p.exists(), "부모 디렉토리가 없어도 만들어야 한다"
        assert cli._load_segments(p) == segs
        assert json.loads(p.read_text(encoding="utf-8"))[0]["text"] == "안녕하세요"

    def test_빈_전사본은_거부한다(self, tmp_path):
        import cli
        import pytest

        p = tmp_path / "empty.json"
        p.write_text("[]", encoding="utf-8")
        with pytest.raises(SystemExit):
            cli._load_segments(p)


class TestRenderStillWorksWithoutSummary:
    def test_요약이_없어도_전사만으로_렌더된다(self):
        # 1차 저장(전사까지)이 이 경로를 탄다. 여기가 깨지면 중간 저장이 무의미해진다.
        md = render_markdown([{"start": 0.0, "end": 1.0, "text": "안녕하세요"}],
                             {"title": "t"}, summary=None, attendees=["홍길동"])
        assert "안녕하세요" in md
        assert "참석 (1명): 홍길동" in md
        # 요약이 없으면 "## 전문" 헤더 없이 본문만 나온다(기존 동작). 섹션이 하나뿐이라
        # 굳이 제목을 달지 않는다.
        assert "## 요약" not in md

    def test_요약이_있으면_섹션이_붙는다(self):
        s = Summary(overview="개요", decisions=["결정1"],
                    action_items=[{"owner": "", "task": "할 일", "due": ""}])
        md = render_markdown([{"start": 0.0, "end": 1.0, "text": "안녕"}], {"title": "t"},
                             summary=s)
        assert "## 요약" in md and "## 결정사항" in md and "## 액션아이템" in md
