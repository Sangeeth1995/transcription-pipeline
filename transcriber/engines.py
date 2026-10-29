"""Speech-to-text engines behind one small interface, so the model is swappable.

Every engine takes 16 kHz mono float32 audio for ONE chunk and returns segments whose
timestamps are relative to that chunk. The pipeline adds the chunk offset.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import Settings


@dataclass
class RawSegment:
    start: float
    end: float
    text: str
    confidence: float | None = None        # 0..1, from the decoder's average log-prob
    no_speech_prob: float | None = None
    extra: dict = field(default_factory=dict)


class Engine(Protocol):
    name: str
    model_name: str
    target_chunk_s: float   # preferred chunk length
    max_chunk_s: float      # hard limit per chunk

    def transcribe(self, audio: np.ndarray, sample_rate: int,
                   language: str | None) -> tuple[list[RawSegment], str | None]: ...


class FasterWhisperEngine:
    """Whisper on CTranslate2: ~4x faster than openai-whisper, int8 on CPU, float16 on GPU."""

    name = "faster-whisper"
    # faster-whisper handles any length internally; we still chunk (5-10 min) for
    # progress, memory, retries and parallelism.
    target_chunk_s = 300.0
    max_chunk_s = 600.0

    def __init__(self, settings: Settings):
        from faster_whisper import WhisperModel   # imported lazily: optional dependency

        self.model_name = settings.model
        self.model = WhisperModel(settings.model, device=settings.device,
                                  compute_type=settings.compute_type,
                                  cpu_threads=settings.threads)

    def transcribe(self, audio, sample_rate, language):
        segments, info = self.model.transcribe(
            audio,
            language=language,                 # None = auto-detect
            beam_size=5,
            vad_filter=True,                   # Silero VAD: skip non-speech inside the chunk
            condition_on_previous_text=False,  # stops repetition loops spreading
        )
        out = [
            RawSegment(start=s.start, end=s.end, text=s.text,
                       confidence=math.exp(s.avg_logprob), no_speech_prob=s.no_speech_prob)
            for s in segments                  # generator: decoding happens here
        ]
        return out, info.language


class SherpaOnnxWhisperEngine:
    """Whisper exported to ONNX, run with sherpa-onnx. Fully offline, tiny install."""

    name = "sherpa-onnx"
    # Whisper's encoder sees at most 30 s, so chunks must stay below that.
    target_chunk_s = 20.0
    max_chunk_s = 28.0

    def __init__(self, settings: Settings):
        import sherpa_onnx

        d = Path(settings.model)
        pick = lambda pattern: sorted(d.glob(pattern), key=lambda p: "int8" not in p.name)[0]
        self.model_name = d.name
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=str(pick("*encoder*.onnx")), decoder=str(pick("*decoder*.onnx")),
            tokens=str(pick("*tokens.txt")), language="", task="transcribe",
            num_threads=settings.threads, enable_segment_timestamps=True,
        )

    def transcribe(self, audio, sample_rate, language):
        stream = self.recognizer.create_stream()
        stream.accept_waveform(sample_rate, audio)
        self.recognizer.decode_stream(stream)
        r = stream.result
        if r.segment_texts:
            segs = [RawSegment(start=t, end=t + d, text=txt)
                    for t, d, txt in zip(r.segment_timestamps, r.segment_durations, r.segment_texts)]
        elif r.text.strip():
            segs = [RawSegment(start=0.0, end=len(audio) / sample_rate, text=r.text)]
        else:
            segs = []
        return segs, (r.lang or None)


def load_engine(settings: Settings) -> Engine:
    if settings.engine == "faster-whisper":
        return FasterWhisperEngine(settings)
    if settings.engine == "sherpa-onnx":
        return SherpaOnnxWhisperEngine(settings)
    raise ValueError(f"Unknown STT_ENGINE {settings.engine!r}")
