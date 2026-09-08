"""도메인 용어 후처리 교정.

Whisper가 IT/보안 전문 용어를 소리 나는 대로 오인식하는 것을 사전 기반으로 바로잡는다.
파인튜닝 없이도 실효가 있어, 가장 비용 대비 효과가 큰 단계다.
"""

import re

from .config import load_corrections

# 영문·숫자로만 된 키는 단어 경계를 요구한다.
# 이것이 없으면 UR -> URL 규칙이 URL을 URLL로, during을 dURLing으로 만든다.
_ASCII_KEY = re.compile(r"^[A-Za-z0-9._-]+$")


class Corrector:
    """용어 사전을 컴파일해 두고 반복 적용한다."""

    def __init__(self, corrections: dict | None = None):
        self._corrections = corrections if corrections is not None else load_corrections()
        # 긴 표현부터 치환해야 부분 매칭이 긴 표현을 깨뜨리지 않는다.
        self._patterns = [
            (re.compile(self._pattern_for(wrong), re.IGNORECASE), correct)
            for wrong, correct in sorted(
                self._corrections.items(), key=lambda kv: len(kv[0]), reverse=True
            )
        ]

    @staticmethod
    def _pattern_for(wrong: str) -> str:
        """영문 키에만 경계를 건다.

        \\b는 못 쓴다. 한글 음절도 \\w라서 "UR을"의 R과 을 사이에 경계가 서지 않는다.
        영문 키는 영문/숫자가 앞뒤에 붙는 경우만 막는다. 한글 조사가 붙은 형태는 그대로 잡힌다.

        한글 키는 앞쪽만 막는다. 한국어는 어절이 띄어쓰기로 갈리고 조사는 뒤에 붙으므로,
        뒤를 막으면 "바스는"을 놓치고, 앞을 안 막으면 "자바스크립트"가 "자OpenVAS크립트"가 된다.
        """
        escaped = re.escape(wrong)
        if _ASCII_KEY.match(wrong):
            return rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])"
        return rf"(?<![가-힣]){escaped}"

    def apply(self, text: str) -> str:
        # 치환값을 콜러블로 넘긴다 — 문자열로 넘기면 re가 '\1'·'\g<..>' 같은 백참조로
        # 해석해, 로컬 사전(corrections.local.json)에 역슬래시가 든 표기가 있으면 깨진다.
        for pattern, correct in self._patterns:
            text = pattern.sub(lambda _m, c=correct: c, text)
        return text

    def __len__(self) -> int:
        return len(self._patterns)
