"""sherpa-onnx 화자 분리 배선 검증.

모델을 실제로 돌리는 테스트는 넣지 않는다(17초 오디오에 10초). 여기서 지키는 것은
배선과 실패 경로다. 모델이 없을 때 조용히 이상하게 돌지 않고 경로를 알려주며 멈추는가,
외부 명령을 셸을 거치지 않고 부르는가.
"""

import subprocess
from pathlib import Path

import pytest

from meetscribe import audio, diarize


class TestWavConversion:
    def test_ffmpeg을_인자_리스트로_부른다(self, tmp_path, monkeypatch):
        """파일명에 공백이나 세미콜론이 들어가도 명령으로 읽히면 안 된다."""
        seen = {}

        def fake_run(cmd, **kw):
            seen["cmd"] = cmd
            seen["kw"] = kw
            Path(cmd[-1]).write_bytes(b"RIFF")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(subprocess, "run", fake_run)
        src = tmp_path / "회의; rm -rf x.m4a"
        src.write_bytes(b"x")
        audio.to_wav_16k_mono(src, tmp_path / "o" / "a.wav")

        assert isinstance(seen["cmd"], list), "문자열로 넘기면 셸 해석 여지가 생긴다"
        assert seen["kw"].get("shell") in (None, False)
        assert str(src) in seen["cmd"], "경로가 쪼개지지 않고 한 인자로 들어가야 한다"
        assert "16000" in seen["cmd"] and "-ac" in seen["cmd"]

    def test_ffmpeg이_없으면_안내하고_멈춘다(self, tmp_path, monkeypatch):
        def boom(*a, **kw):
            raise FileNotFoundError()

        monkeypatch.setattr(subprocess, "run", boom)
        with pytest.raises(RuntimeError, match="ffmpeg"):
            audio.to_wav_16k_mono(tmp_path / "a.m4a", tmp_path / "b.wav")

    def test_변환은_실패했는데_성공으로_읽지_않는다(self, tmp_path, monkeypatch):
        # returncode 0이어도 출력 파일이 없으면 실패다. 빈 결과를 성공으로 넘기면
        # 다음 단계가 이유 없이 죽는다.
        monkeypatch.setattr(subprocess, "run",
                            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "", ""))
        with pytest.raises(RuntimeError, match="변환 실패"):
            audio.to_wav_16k_mono(tmp_path / "a.m4a", tmp_path / "b.wav")


class TestModelGate:
    def test_모델이_없으면_경로를_알려주고_멈춘다(self, tmp_path):
        missing = tmp_path / "없는모델.onnx"
        with pytest.raises(RuntimeError) as e:
            diarize.diarize_onnx(tmp_path / "a.wav", num_speakers=2,
                                 seg_model=missing, emb_model=missing)
        assert "없는모델.onnx" in str(e.value), "어느 파일이 없는지 말해야 한다"

    def test_기본_모델_경로는_레포_안_models다(self):
        # 모델은 gitignore 대상이라 clone 직후엔 없다. 경로 규약만 고정해 둔다.
        assert diarize.SEG_MODEL.parent.parent.name == "models"
        assert diarize.EMB_MODEL.name == "eres2net.onnx", (
            "CAM++는 발화량 적은 화자를 흡수해 실측에서 탈락했다"
        )


class TestAssignSpeakers:
    def test_겹침이_가장_큰_화자를_고른다(self):
        turns = [{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
                 {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_01"}]
        segs = [{"start": 4.0, "end": 9.0, "text": "걸친 발언"}]
        # 4~5는 00, 5~9는 01 이므로 01이 이긴다
        assert diarize.assign_speakers(segs, turns)[0]["speaker"] == "SPEAKER_01"

    def test_화자_구간이_없으면_speaker가_None이다(self):
        # 키를 빼는 게 아니라 None을 넣는다. 렌더러가 .get("speaker")로 보므로
        # 화자 접두사 없이 그려지고, 화자 분리가 실패해도 회의록은 나온다.
        segs = [{"start": 0.0, "end": 1.0, "text": "발언"}]
        assert diarize.assign_speakers(segs, [])[0]["speaker"] is None
