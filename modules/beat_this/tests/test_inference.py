import numpy as np
import torch
from pathlib import Path
import subprocess
import sys

from beat_this.inference import File2Beats, Audio2Frames
from beat_this.preprocessing import load_audio


_TEST_AUDIO = Path(__file__).resolve().parent / "It Don't Mean A Thing - Kings of Swing.mp3"


def test_File2Beat():
    f2b = File2Beats()
    beat, downbeat = f2b(_TEST_AUDIO)
    assert isinstance(beat, np.ndarray)
    assert isinstance(downbeat, np.ndarray)


def test_Audio2Frames():
    a2f = Audio2Frames()
    audio, sr = load_audio(_TEST_AUDIO, dtype="float32")
    beat, downbeat = a2f(audio, sr)
    assert isinstance(beat, torch.Tensor)
    assert isinstance(downbeat, torch.Tensor)


def test_beat_to_osu_dbn_fallback_produces_output(tmp_path):
    script = Path(__file__).resolve().parents[1] / "beat_to_osu.py"
    out = tmp_path / "timing_points.txt"

    r = subprocess.run(
        [
            sys.executable,
            str(script),
            str(_TEST_AUDIO),
            "--device",
            "cpu",
            "--checkpoint",
            "final0",
            "--postprocess",
            "dbn",
            "--method",
            "gridfit",
            "-o",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    assert out.exists()
    assert out.stat().st_size > 0
    assert "Detected 0 beats" not in (r.stdout + r.stderr)
