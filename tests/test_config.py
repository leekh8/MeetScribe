"""config.load_corrections — 기본 사전 + 로컬 병합."""

import json

from meetscribe import config


def test_defaults_present(monkeypatch, tmp_path):
    # 로컬 사전 경로를 존재하지 않는 곳으로 돌려 기본 사전만 검증
    monkeypatch.setattr(config, "LOCAL_DICT_PATH", tmp_path / "none.json")
    corrections = config.load_corrections()
    assert corrections["시엠"] == "SIEM"
    assert corrections["나슬"] == "NASL"


def test_local_overrides_and_adds(monkeypatch, tmp_path):
    local = tmp_path / "corrections.local.json"
    local.write_text(json.dumps({"시엠": "OVERRIDE", "새용어": "NEW"}), encoding="utf-8")
    monkeypatch.setattr(config, "LOCAL_DICT_PATH", local)
    corrections = config.load_corrections()
    assert corrections["시엠"] == "OVERRIDE"   # 로컬이 기본을 덮어씀
    assert corrections["새용어"] == "NEW"       # 로컬 신규 항목 추가


def test_malformed_local_falls_back_to_defaults(monkeypatch, tmp_path, capsys):
    bad = tmp_path / "corrections.local.json"
    bad.write_text("{ this is not json", encoding="utf-8")
    monkeypatch.setattr(config, "LOCAL_DICT_PATH", bad)
    corrections = config.load_corrections()
    assert corrections["시엠"] == "SIEM"        # 깨진 로컬이어도 기본 사전은 유효
