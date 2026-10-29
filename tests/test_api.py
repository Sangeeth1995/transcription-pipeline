import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.conftest import SAMPLES
from transcriber.api import create_app
from transcriber.config import Settings


@pytest.fixture
def client(fake_engine, settings):
    with TestClient(create_app(engine=fake_engine, settings=settings)) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "engine": "fake", "model": "fake-1"}


def test_transcribe_returns_segments_with_timestamps(client, tone_wav):
    r = client.post("/transcribe", files={"file": ("tone.wav", tone_wav.read_bytes(), "audio/wav")})
    assert r.status_code == 200
    body = r.json()
    assert len(body["segments"]) == 2
    assert {"id", "start", "end", "text", "confidence", "flags"} <= body["segments"][0].keys()
    assert body["metadata"]["source"]["sample_rate"] == 44100


def test_srt_output(client, tone_wav):
    r = client.post("/transcribe?format=srt", files={"file": ("t.wav", tone_wav.read_bytes())})
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/x-subrip")
    assert r.text.startswith("1\n00:00:00,")


def test_rejects_non_audio_and_empty(client):
    r = client.post("/transcribe", files={"file": ("x.mp3", b"definitely not audio")})
    assert r.status_code == 415
    assert client.post("/transcribe", files={"file": ("x.wav", b"")}).status_code == 400


def test_long_audio_is_redirected_to_jobs(fake_engine, tone_wav):
    s = Settings(engine="fake", model="fake", sync_limit_s=5)          # tone is 6 s
    with TestClient(create_app(engine=fake_engine, settings=s)) as c:
        r = c.post("/transcribe", files={"file": ("t.wav", tone_wav.read_bytes())})
        assert r.status_code == 413 and "/jobs" in r.json()["detail"]

        job = c.post("/jobs", files={"file": ("t.wav", tone_wav.read_bytes())})
        assert job.status_code == 202
        url = job.json()["status_url"]
        for _ in range(100):
            j = c.get(url).json()
            if j["status"] in ("done", "failed"):
                break
            time.sleep(0.05)
        assert j["status"] == "done", j
        assert j["progress"] == {"done": 2, "total": 2}
        assert len(j["result"]["segments"]) == 2
        assert c.get(url + "?format=vtt").text.startswith("WEBVTT")


MODEL_DIR = os.getenv("SHERPA_MODEL_DIR")


@pytest.mark.skipif(not MODEL_DIR, reason="set SHERPA_MODEL_DIR to run the real-model test")
def test_real_model_end_to_end_mp3():
    from transcriber.engines import SherpaOnnxWhisperEngine
    s = Settings(engine="sherpa-onnx", model=MODEL_DIR)
    with TestClient(create_app(engine=SherpaOnnxWhisperEngine(s), settings=s)) as c:
        mp3 = Path(SAMPLES / "stereo_44k.mp3").read_bytes()
        body = c.post("/transcribe", files={"file": ("s.mp3", mp3)}).json()
    text = body["text"].lower()
    assert "direct consequence of the sin" in text and "blessed soul in heaven" in text
    starts = [seg["start"] for seg in body["segments"]]
    assert starts == sorted(starts) and body["segments"][-1]["end"] <= body["duration"]
