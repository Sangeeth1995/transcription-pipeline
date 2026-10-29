"""Long-audio handling: split the normalised WAV into chunks at natural pauses.

Why chunk at all (even though Whisper slides a 30 s window internally)?
  * bounded memory: each chunk is read from disk on its own;
  * progress reporting and partial results for long jobs;
  * fault isolation: one bad chunk is retried or skipped, not the whole file;
  * the chunks are independent, so they can be fanned out to several workers/GPUs;
  * silence is skipped entirely, which saves compute and avoids Whisper's habit of
    "hallucinating" text (e.g. "Thanks for watching!") on long silent stretches.

Cuts are placed in silences found by ffmpeg's `silencedetect`, so words are never split.
If a stretch of speech has no pause at all (music, crowd noise), we fall back to cutting
at the quietest 50 ms frame near the limit.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np

from .audio import read_segment

_START = re.compile(r"silence_start:\s*(-?[\d.]+)")
_END = re.compile(r"silence_end:\s*(-?[\d.]+)")


def detect_silences(wav: str | Path, duration: float, noise_db: float = -35.0,
                    min_silence_s: float = 0.4) -> list[tuple[float, float]]:
    """Return [(start, end), ...] of silent stretches, using ffmpeg's streaming filter."""
    proc = subprocess.run([
        "ffmpeg", "-nostdin", "-hide_banner", "-i", str(wav),
        "-af", f"silencedetect=noise={noise_db}dB:d={min_silence_s}",
        "-f", "null", "-",
    ], capture_output=True, text=True)
    silences: list[tuple[float, float]] = []
    start: float | None = None
    for line in proc.stderr.splitlines():
        if (m := _START.search(line)):
            start = max(0.0, float(m.group(1)))
        elif (m := _END.search(line)) and start is not None:
            silences.append((start, min(duration, float(m.group(1)))))
            start = None
    if start is not None:                     # file ends in silence
        silences.append((start, duration))
    return silences


def quietest_point(wav: str | Path, lo: float, hi: float, frame_s: float = 0.05) -> float:
    """Time of the lowest-energy frame in [lo, hi] - a safe place for a forced cut."""
    audio = read_segment(wav, lo, hi)
    n = int(frame_s * 16_000)
    if len(audio) < 2 * n:
        return hi
    frames = audio[: len(audio) // n * n].reshape(-1, n)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    return lo + (int(rms.argmin()) + 0.5) * frame_s


def plan_chunks(duration: float, silences: list[tuple[float, float]], target_s: float,
                max_s: float, pad_s: float = 0.15, min_speech_s: float = 0.2,
                cut_at=None) -> list[tuple[float, float]]:
    """Turn silences into chunk boundaries.

    1. speech regions = everything that is not silence (tiny blips dropped);
    2. each region is padded slightly so the first/last word is not clipped;
    3. regions longer than `max_s` are force-split (at `cut_at(lo, hi)` if given);
    4. neighbouring regions are packed together while the chunk stays <= `target_s`.
    Returns [] when the file is entirely silent.
    """
    regions: list[tuple[float, float]] = []
    cursor = 0.0
    for s, e in sorted(silences):
        if s - cursor >= min_speech_s:
            regions.append((cursor, s))
        cursor = max(cursor, e)
    if duration - cursor >= min_speech_s:
        regions.append((cursor, duration))

    padded: list[tuple[float, float]] = []
    for s, e in regions:
        s, e = max(0.0, s - pad_s), min(duration, e + pad_s)
        if padded and s < padded[-1][1]:
            s = padded[-1][1]
        padded.append((s, e))

    pieces: list[tuple[float, float]] = []
    for s, e in padded:
        while e - s > max_s:
            lo, hi = s + 0.8 * max_s, s + max_s
            cut = cut_at(lo, hi) if cut_at else hi
            pieces.append((s, cut))
            s = cut
        pieces.append((s, e))

    chunks: list[tuple[float, float]] = []
    for s, e in pieces:
        if chunks and e - chunks[-1][0] <= target_s:
            chunks[-1] = (chunks[-1][0], e)
        else:
            chunks.append((s, e))
    return chunks
