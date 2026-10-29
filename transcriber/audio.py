"""Audio I/O: probe any container/codec with ffprobe and normalise it with ffmpeg.

Design choice: instead of special-casing WAV, MP3, M4A, OGG, ... we decode *everything*
through ffmpeg into one canonical format (16 kHz, mono, 16-bit PCM WAV). The rest of the
pipeline only ever sees that format, so adding a new input format costs nothing.
Formats are detected from the file's content (ffprobe), never trusted from the extension.
"""
from __future__ import annotations

import json
import subprocess
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


class AudioError(Exception):
    """Raised for files that are not decodable audio (maps to HTTP 415)."""


@dataclass
class AudioInfo:
    format: str          # container, e.g. "mp3", "wav", "mov,mp4,m4a,3gp,3g2,mj2"
    codec: str           # e.g. "mp3", "pcm_s16le", "aac", "opus"
    sample_rate: int
    channels: int
    duration: float      # seconds

    def as_dict(self) -> dict:
        return asdict(self)


def _run(cmd: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def probe(path: str | Path) -> AudioInfo:
    """Inspect the file's real content. Works for audio-only files and for video with audio."""
    proc = _run([
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", "-select_streams", "a", str(path),
    ], timeout=60)
    if proc.returncode != 0:
        raise AudioError(f"Not a readable media file: {proc.stderr.strip()[:200]}")
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams") or []
    if not streams:
        raise AudioError("File contains no audio stream")
    s = streams[0]
    duration = float(s.get("duration") or data.get("format", {}).get("duration") or 0.0)
    return AudioInfo(
        format=data.get("format", {}).get("format_name", "unknown"),
        codec=s.get("codec_name", "unknown"),
        sample_rate=int(s.get("sample_rate") or 0),
        channels=int(s.get("channels") or 0),
        duration=duration,
    )


def normalize(src: str | Path, dst: str | Path, sample_rate: int = 16_000) -> Path:
    """Decode any input to 16 kHz mono 16-bit PCM WAV (first audio track, video dropped)."""
    dst = Path(dst)
    proc = _run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        "-map", "0:a:0",          # first audio track only
        "-vn", "-sn", "-dn",      # drop video, subtitles, data
        "-ac", "1",               # down-mix to mono
        "-ar", str(sample_rate),  # resample (8 kHz phone audio, 44.1/48 kHz music, ...)
        "-c:a", "pcm_s16le", "-f", "wav", str(dst),
    ])
    if proc.returncode != 0 or not dst.exists() or dst.stat().st_size <= 44:
        raise AudioError(f"Could not decode audio: {proc.stderr.strip()[:200]}")
    return dst


def wav_duration(path: str | Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def read_segment(path: str | Path, start: float, end: float) -> np.ndarray:
    """Read [start, end) seconds of a mono 16-bit WAV as float32 in [-1, 1].

    Only the requested window is read, so memory stays flat even for multi-hour files.
    """
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        first = max(0, int(start * sr))
        count = max(0, int(end * sr) - first)
        w.setpos(min(first, w.getnframes()))
        frames = w.readframes(count)
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
