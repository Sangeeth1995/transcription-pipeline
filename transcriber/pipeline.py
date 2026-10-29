"""The pipeline: probe -> normalise -> plan chunks -> transcribe each chunk -> post-process."""
from __future__ import annotations

import logging
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .audio import AudioError, normalize, probe, read_segment, wav_duration
from .chunking import detect_silences, plan_chunks, quietest_point
from .config import Settings
from .engines import Engine, RawSegment
from .postprocess import Segment, full_text, postprocess

log = logging.getLogger("transcriber")
Progress = Callable[[int, int], None]


@dataclass
class Transcript:
    text: str
    segments: list[Segment]
    language: str | None
    duration: float
    metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"text": self.text, "language": self.language, "duration": self.duration,
                "segments": [s.as_dict() for s in self.segments], "metadata": self.metadata}


def transcribe_file(path: str | Path, engine: Engine, settings: Settings,
                    language: str | None = None, progress: Progress | None = None,
                    lock: threading.Semaphore | None = None, retries: int = 1) -> Transcript:
    t0 = time.perf_counter()
    source = probe(path)                                  # raises AudioError if not audio
    if source.duration > settings.max_duration_s:
        raise AudioError(f"Audio is {source.duration:.0f}s; limit is {settings.max_duration_s:.0f}s")

    with tempfile.TemporaryDirectory(prefix="stt-") as tmp:
        wav = normalize(path, Path(tmp) / "audio.wav", settings.sample_rate)
        duration = wav_duration(wav)
        silences = detect_silences(wav, duration, settings.silence_db, settings.min_silence_s)
        chunks = plan_chunks(duration, silences, engine.target_chunk_s, engine.max_chunk_s,
                             cut_at=lambda lo, hi: quietest_point(wav, lo, hi))
        log.info("%s: %.1fs audio -> %d chunks", Path(path).name, duration, len(chunks))

        raw: list[RawSegment] = []
        languages: list[str] = []
        failed: list[dict] = []
        for i, (start, end) in enumerate(chunks):
            audio = read_segment(wav, start, end)
            for attempt in range(retries + 1):
                try:
                    if lock:
                        with lock:
                            segs, lang = engine.transcribe(audio, settings.sample_rate, language)
                    else:
                        segs, lang = engine.transcribe(audio, settings.sample_rate, language)
                    break
                except Exception as exc:                  # keep going; report the gap
                    log.warning("chunk %d attempt %d failed: %s", i, attempt + 1, exc)
                    if attempt == retries:
                        failed.append({"chunk": i, "start": round(start, 3),
                                       "end": round(end, 3), "error": str(exc)[:200]})
                        segs, lang = [], None
            for s in segs:                                 # chunk-relative -> absolute time
                s.start += start
                s.end += start
            raw.extend(segs)
            if lang:
                languages.append(lang)
            if progress:
                progress(i + 1, len(chunks))

    segments = postprocess(raw, duration)
    elapsed = time.perf_counter() - t0
    return Transcript(
        text=full_text(segments),
        segments=segments,
        language=language or (max(set(languages), key=languages.count) if languages else None),
        duration=round(duration, 3),
        metadata={
            "engine": engine.name, "model": engine.model_name,
            "source": source.as_dict(),
            "chunks": len(chunks),
            "speech_seconds": round(sum(e - s for s, e in chunks), 3),
            "failed_chunks": failed,
            "processing_seconds": round(elapsed, 3),
            "real_time_factor": round(elapsed / duration, 3) if duration else None,
        },
    )
