"""Command line: python -m transcriber.cli meeting.mp3 --format srt -o meeting.srt"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys

from .audio import AudioError
from .config import get_settings
from .engines import load_engine
from .formats import RENDERERS
from .pipeline import transcribe_file


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Transcribe an audio or video file.")
    p.add_argument("path")
    p.add_argument("--engine", help="faster-whisper | sherpa-onnx (default from STT_ENGINE)")
    p.add_argument("--model", help="model size or model directory (default from STT_MODEL)")
    p.add_argument("--language", help="ISO code, e.g. en; omit to auto-detect")
    p.add_argument("--format", choices=["json", "srt", "vtt", "txt"], default="json")
    p.add_argument("-o", "--output", help="write to this file instead of stdout")
    a = p.parse_args(argv)

    settings = get_settings()
    overrides = {k: v for k, v in {"engine": a.engine, "model": a.model}.items() if v}
    settings = dataclasses.replace(settings, **overrides)
    engine = load_engine(settings)

    def progress(done, total):
        print(f"\rchunk {done}/{total}", end="", file=sys.stderr, flush=True)

    try:
        result = transcribe_file(a.path, engine, settings, a.language, progress)
    except AudioError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(file=sys.stderr)
    text = (json.dumps(result.as_dict(), indent=2, ensure_ascii=False) if a.format == "json"
            else RENDERERS[a.format][0](result))
    if a.output:
        with open(a.output, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
