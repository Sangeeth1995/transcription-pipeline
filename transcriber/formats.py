"""Output formats: JSON (for systems), SRT/VTT (for players), plain text (for people)."""
from __future__ import annotations

from .pipeline import Transcript


def _ts(seconds: float, sep: str) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_srt(t: Transcript) -> str:
    return "\n".join(
        f"{i}\n{_ts(s.start, ',')} --> {_ts(s.end, ',')}\n{s.text}\n"
        for i, s in enumerate(t.segments, start=1)
    )


def to_vtt(t: Transcript) -> str:
    body = "\n".join(f"{_ts(s.start, '.')} --> {_ts(s.end, '.')}\n{s.text}\n" for s in t.segments)
    return "WEBVTT\n\n" + body


def to_txt(t: Transcript) -> str:
    return "\n".join(f"[{_ts(s.start, '.')} - {_ts(s.end, '.')}] {s.text}" for s in t.segments) + "\n"


RENDERERS = {"srt": (to_srt, "application/x-subrip"),
             "vtt": (to_vtt, "text/vtt"),
             "txt": (to_txt, "text/plain")}
