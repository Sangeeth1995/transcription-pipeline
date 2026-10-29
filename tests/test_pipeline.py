import sys
import types

import numpy as np
import pytest

from transcriber.audio import AudioError, normalize, probe, read_segment, wav_duration
from transcriber.chunking import detect_silences, plan_chunks, quietest_point
from transcriber.config import Settings
from transcriber.engines import RawSegment
from transcriber.formats import to_srt, to_vtt
from transcriber.pipeline import transcribe_file
from transcriber.postprocess import postprocess


# ---------- audio formats ----------

def test_any_format_is_normalised_to_16k_mono(tone_wav, tmp_path):
    from tests.conftest import ffmpeg
    mp3 = tmp_path / "tone.mp3"
    ffmpeg("-i", str(tone_wav), str(mp3))
    for src in (tone_wav, mp3):
        info = probe(src)
        assert info.sample_rate == 44100 and info.channels == 2
        out = normalize(src, tmp_path / f"{src.stem}-{src.suffix[1:]}.wav")
        n = probe(out)
        assert (n.sample_rate, n.channels, n.codec) == (16000, 1, "pcm_s16le")
        assert abs(wav_duration(out) - 6.0) < 0.1


def test_format_detected_from_content_not_extension(tone_wav, tmp_path):
    disguised = tmp_path / "really_a_wav.mp3"
    disguised.write_bytes(tone_wav.read_bytes())
    assert probe(disguised).codec == "pcm_s16le"


def test_non_audio_is_rejected(tmp_path):
    bad = tmp_path / "notes.wav"
    bad.write_text("not audio")
    with pytest.raises(AudioError):
        probe(bad)


# ---------- long audio / chunking ----------

def test_silence_detection_finds_the_gap(tone_wav, tmp_path):
    wav = normalize(tone_wav, tmp_path / "n.wav")
    silences = detect_silences(wav, wav_duration(wav))
    assert len(silences) == 1
    s, e = silences[0]
    assert abs(s - 3.0) < 0.1 and abs(e - 4.0) < 0.1


def test_chunks_cut_in_silence_and_skip_it():
    # speech 0-10, silence 10-12, speech 12-20, silence 20-30 (trailing)
    chunks = plan_chunks(30.0, [(10, 12), (20, 30)], target_s=15, max_s=28, pad_s=0)
    assert chunks == [(0.0, 10), (12, 20)]          # silence never sent to the model


def test_regions_are_packed_up_to_target():
    chunks = plan_chunks(30.0, [(5, 6), (11, 12), (25, 26)], target_s=12, max_s=28, pad_s=0)
    assert chunks == [(0.0, 11), (12, 25), (26, 30.0)]


def test_long_unbroken_speech_is_force_split_under_max():
    chunks = plan_chunks(100.0, [], target_s=20, max_s=28, pad_s=0)
    assert all(e - s <= 28 for s, e in chunks)
    assert chunks[0][0] == 0 and chunks[-1][1] == 100.0
    assert all(a[1] == b[0] for a, b in zip(chunks, chunks[1:]))   # contiguous, no gaps


def test_all_silence_gives_no_chunks():
    assert plan_chunks(10.0, [(0, 10)], target_s=20, max_s=28) == []


def test_forced_cut_lands_on_quietest_frame(tmp_path):
    from tests.conftest import ffmpeg
    # loud tone with a 0.1 s dip at 9.5 s - too short for silencedetect, ideal forced cut
    src = tmp_path / "dip.wav"
    ffmpeg("-f", "lavfi", "-i", "sine=f=300:sample_rate=16000:duration=12",
           "-af", "volume=enable='between(t,9.5,9.6)':volume=0.01", str(src))
    assert abs(quietest_point(src, 8.0, 10.0) - 9.55) < 0.06


# ---------- pipeline + timestamps ----------

def test_timestamps_are_absolute_across_chunks(tone_wav, fake_engine, settings):
    t = transcribe_file(tone_wav, fake_engine, settings)
    # tone 0-3 s, silence 3-4 s, tone 4-6 s -> two chunks (target 2.5 s)
    assert t.metadata["chunks"] == 2
    first, second = t.segments
    assert first.start < 0.2 and abs(first.end - 3.0) < 0.3
    assert abs(second.start - 4.0) < 0.3 and abs(second.end - 6.0) < 0.2
    assert t.language == "en" and t.text == "chunk 1 chunk 2"


def test_failed_chunk_is_reported_not_fatal(tone_wav, settings):
    class Flaky:
        name, model_name, target_chunk_s, max_chunk_s = "flaky", "x", 2.5, 4.0
        calls = 0

        def transcribe(self, audio, sr, language):
            self.calls += 1
            if self.calls <= 2:                       # chunk 0 fails twice (1 retry)
                raise RuntimeError("GPU out of memory")
            return [RawSegment(0, 1, "ok")], "en"

    t = transcribe_file(tone_wav, Flaky(), settings)
    assert [f["chunk"] for f in t.metadata["failed_chunks"]] == [0]
    assert t.text == "ok"


# ---------- post-processing ----------

def test_postprocess_cleans_orders_and_flags():
    raw = [
        RawSegment(5, 7, "  second   part ", confidence=0.2),
        RawSegment(0, 5, "First part.", confidence=0.9),
        RawSegment(7, 9, "second part", confidence=0.9),        # Whisper repetition
        RawSegment(9, 10, "Thanks for watching!", confidence=0.1, no_speech_prob=0.9),
        RawSegment(10, 99, "tail", confidence=0.8),
    ]
    segs = postprocess(raw, duration=12.0)
    assert [s.text for s in segs] == ["First part.", "second part", "tail"]
    assert segs[1].flags == ["low_confidence", "repeated"] and segs[1].end == 9
    assert segs[-1].end == 12.0                                   # clamped to file length
    assert [s.id for s in segs] == [0, 1, 2]


def test_subtitle_formats(tone_wav, fake_engine, settings):
    t = transcribe_file(tone_wav, fake_engine, settings)
    srt = to_srt(t)
    assert srt.startswith("1\n00:00:00,") and " --> 00:00:0" in srt
    assert to_vtt(t).startswith("WEBVTT\n\n00:00:00.")


# ---------- faster-whisper adapter (model mocked: no download needed) ----------

def test_faster_whisper_adapter_maps_segments(monkeypatch):
    seg = types.SimpleNamespace(start=0.0, end=2.5, text=" Hello there", avg_logprob=-0.1,
                                no_speech_prob=0.01)

    class FakeWhisperModel:
        def __init__(self, *a, **k):
            self.kwargs = k

        def transcribe(self, audio, **kw):
            assert kw["vad_filter"] is True and kw["condition_on_previous_text"] is False
            return iter([seg]), types.SimpleNamespace(language="en")

    monkeypatch.setitem(sys.modules, "faster_whisper",
                        types.SimpleNamespace(WhisperModel=FakeWhisperModel))
    from transcriber.engines import FasterWhisperEngine
    eng = FasterWhisperEngine(Settings(engine="faster-whisper", model="small"))
    segs, lang = eng.transcribe(np.zeros(16000, dtype=np.float32), 16000, None)
    assert lang == "en" and segs[0].text == " Hello there"
    assert segs[0].confidence == pytest.approx(0.905, abs=1e-3)
