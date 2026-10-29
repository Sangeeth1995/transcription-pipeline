"""Clean raw engine output into segments that downstream systems can trust."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .engines import RawSegment

LOW_CONFIDENCE = 0.40     # exp(avg_logprob) below this -> flag for human review
NO_SPEECH = 0.60          # Whisper's own "this is probably not speech" probability


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    confidence: float | None = None
    flags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def postprocess(raw: list[RawSegment], duration: float) -> list[Segment]:
    """Order, clamp, de-duplicate and flag segments. Nothing is silently invented."""
    out: list[Segment] = []
    last_end = 0.0
    for r in sorted(raw, key=lambda s: s.start):
        text = _clean(r.text)
        if not text:
            continue
        # Whisper sometimes emits text for silence; drop it only when the model is
        # confident it is not speech AND the decode itself was weak.
        confidence = r.confidence if r.confidence is not None else 1.0
        no_speech = (
            r.no_speech_prob
            if r.no_speech_prob is not None
            else 0.0
        )
        if no_speech > NO_SPEECH and confidence < LOW_CONFIDENCE:
            continue
        # if (r.no_speech_prob or 0) > NO_SPEECH and (r.confidence or 1) < LOW_CONFIDENCE:
        #     continue
        start = round(min(max(r.start, last_end), duration), 3)   # monotonic, in range
        end = round(min(max(r.end, start), duration), 3)
        flags: list[str] = []
        if out and text.lower() == out[-1].text.lower():          # repetition loop
            out[-1].end = end
            if "repeated" not in out[-1].flags:
                out[-1].flags.append("repeated")
            continue
        if r.confidence is not None and r.confidence < LOW_CONFIDENCE:
            flags.append("low_confidence")
        out.append(Segment(id=len(out), start=start, end=end, text=text,
                           confidence=None if r.confidence is None else round(r.confidence, 3),
                           flags=flags))
        last_end = end
    return out


def full_text(segments: list[Segment]) -> str:
    return " ".join(s.text for s in segments)
