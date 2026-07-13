"""경로·기본값·도메인 용어 사전 로딩."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = ROOT / "out"
LOCAL_DICT_PATH = ROOT / "corrections.local.json"

DEFAULT_MODEL = "medium"
DEFAULT_LANGUAGE = "ko"

# 공개된 IT/보안 제품·기술 용어만 수록한다.
# 조직·고객사 특화 용어는 corrections.local.json(gitignore)에 둔다 — 조직 정보 유출 방지.
DEFAULT_CORRECTIONS = {
    "시엠": "SIEM",
    "에스아이이엠": "SIEM",
    "나슬": "NASL",
    "뉴클리": "Nuclei",
    "누클리": "Nuclei",
    "누클레이": "Nuclei",
    "씨브이이": "CVE",
    "씨비이": "CVE",
    "스플렁크": "Splunk",
    "오픈 바스": "OpenVAS",
    "오픈바스": "OpenVAS",
    "이에스엑스아이": "ESXi",
    "큐카우": "QCOW2",
    "큐카우투": "QCOW2",
    "쿠버네티스": "Kubernetes",
    "도커": "Docker",
    "깃허브": "GitHub",
    "에이피아이": "API",
}


def load_corrections() -> dict:
    """기본 사전 + 로컬 사전(있으면) 병합. 로컬이 기본을 덮어쓴다."""
    corrections = dict(DEFAULT_CORRECTIONS)
    if LOCAL_DICT_PATH.exists():
        try:
            local = json.loads(LOCAL_DICT_PATH.read_text(encoding="utf-8"))
            if isinstance(local, dict):
                corrections.update(local)
        except (json.JSONDecodeError, OSError) as e:
            print(f"경고: {LOCAL_DICT_PATH.name} 로드 실패 ({e}) — 기본 사전만 사용")
    return corrections
