"""diarize.assign_speakers — 겹침 기반 화자 배정 (순수 함수)."""

from meetscribe.diarize import assign_speakers


def test_assigns_speaker_with_max_overlap():
    segs = [{"start": 0, "end": 10}]
    turns = [
        {"start": 0, "end": 6, "speaker": "A"},   # 6초 겹침
        {"start": 6, "end": 10, "speaker": "B"},  # 4초 겹침
    ]
    assert assign_speakers(segs, turns)[0]["speaker"] == "A"


def test_no_overlap_leaves_speaker_none():
    segs = [{"start": 100, "end": 110}]
    turns = [{"start": 0, "end": 10, "speaker": "A"}]
    assert assign_speakers(segs, turns)[0]["speaker"] is None


def test_empty_turns_leaves_speaker_none():
    segs = [{"start": 0, "end": 5}]
    assert assign_speakers(segs, [])[0]["speaker"] is None
