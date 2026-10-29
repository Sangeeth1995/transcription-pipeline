import subprocess
from pathlib import Path

import numpy as np
import pytest

from transcriber.config import Settings
from transcriber.engines import RawSegment

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


@pytest.fixture
def tone_wav(tmp_path):
    """3 s of 440 Hz tone, 1 s silence, 2 s tone at 44.1 kHz stereo - a stand-in for speech."""
    a, gap, b = tmp_path / "a.wav", tmp_path / "gap.wav", tmp_path / "b.wav"
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=3", "-ac", "2", str(a))
    ffmpeg("-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", "1", str(gap))
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=2", "-ac", "2", str(b))
    out = tmp_path / "tone.wav"
    ffmpeg("-i", str(a), "-i", str(gap), "-i", str(b),
           "-filter_complex", "[0:a][1:a][2:a]concat=n=3:v=0:a=1", str(out))
    return out


class FakeEngine:
    """Deterministic engine: one segment per chunk, with the chunk's length in the text."""
    name = "fake"
    model_name = "fake-1"
    target_chunk_s = 2.5
    max_chunk_s = 4.0

    def __init__(self):
        self.calls = 0

    def transcribe(self, audio: np.ndarray, sample_rate: int, language):
        self.calls += 1
        secs = len(audio) / sample_rate
        return [RawSegment(0.0, secs, f" chunk {self.calls} ", confidence=0.9)], language or "en"


@pytest.fixture
def fake_engine():
    return FakeEngine()


@pytest.fixture
def settings():
    return Settings(engine="fake", model="fake", sync_limit_s=60)
