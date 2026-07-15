"""도메인 용어 후처리 교정.

Whisper가 IT/보안 전문 용어를 소리 나는 대로 오인식하는 것을 사전 기반으로 바로잡는다.
파인튜닝 없이도 실효가 있어, 가장 비용 대비 효과가 큰 단계다.
"""

import re

from .config import load_corrections


class Corrector:
    """용어 사전을 컴파일해 두고 반복 적용한다."""

    def __init__(self, corrections: dict | None = None):
        self._corrections = corrections if corrections is not None else load_corrections()
        # 긴 표현부터 치환해야 부분 매칭이 긴 표현을 깨뜨리지 않는다.
        self._patterns = [
            (re.compile(re.escape(wrong), re.IGNORECASE), correct)
            for wrong, correct in sorted(
                self._corrections.items(), key=lambda kv: len(kv[0]), reverse=True
            )
        ]

    def apply(self, text: str) -> str:
        # 치환값을 콜러블로 넘긴다 — 문자열로 넘기면 re가 '\1'·'\g<..>' 같은 백참조로
        # 해석해, 로컬 사전(corrections.local.json)에 역슬래시가 든 표기가 있으면 깨진다.
        for pattern, correct in self._patterns:
            text = pattern.sub(lambda _m, c=correct: c, text)
        return text

    def __len__(self) -> int:
        return len(self._patterns)
