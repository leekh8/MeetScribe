"""경로·기본값·도메인 용어 사전 로딩."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = ROOT / "out"
LOCAL_DICT_PATH = ROOT / "corrections.local.json"
# 오인식이 아닌데 후보로 올라오는 말(인명 등). 사전에 넣을 수 없어 따로 둔다.
LOCAL_IGNORE_PATH = ROOT / "vocab_ignore.local.json"
# 사람 호칭 모음. 한 사람이 여러 이름으로 불리면 호칭마다 따로 후보로 올라온다.
LOCAL_PEOPLE_PATH = ROOT / "people.local.json"

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


def load_ignored() -> set:
    """다시 묻지 않을 말 목록. 오인식이 아니라 교정 대상이 아닌 것들이다."""
    if not LOCAL_IGNORE_PATH.exists():
        return set()
    try:
        data = json.loads(LOCAL_IGNORE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"경고: {LOCAL_IGNORE_PATH.name} 로드 실패 ({e}) - 무시 목록 없이 진행")
        return set()
    return {str(x) for x in data} if isinstance(data, list) else set()


def load_people() -> dict:
    """정식 이름 -> 호칭 목록.

    같은 사람이 "이규해 주임", "규해 주임", "규주"로 불리면 셋 다 다른 토큰이 된다.
    호칭은 교정 대상이 아니므로 사전이 아니라 여기에 모아 후보에서 빼기만 한다.
    """
    if not LOCAL_PEOPLE_PATH.exists():
        return {}
    try:
        data = json.loads(LOCAL_PEOPLE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"경고: {LOCAL_PEOPLE_PATH.name} 로드 실패 ({e}) - 호칭 목록 없이 진행")
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): [str(a) for a in v] if isinstance(v, list) else []
            for k, v in data.items()}
