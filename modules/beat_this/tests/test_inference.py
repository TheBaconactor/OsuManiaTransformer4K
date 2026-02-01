import numpy as np
import torch
from pathlib import Path

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
