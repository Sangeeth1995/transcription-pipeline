"""Runtime settings, read from environment variables so the same image runs anywhere."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    # Engine: "faster-whisper" (default) or "sherpa-onnx" (fully offline ONNX Whisper).
    engine: str = _env("STT_ENGINE", "faster-whisper")
    # faster-whisper: a size ("tiny", "base", "small", "medium", "large-v3") or a local CTranslate2 dir.
    # sherpa-onnx: a directory holding *-encoder*.onnx, *-decoder*.onnx and *-tokens.txt.
    model: str = _env("STT_MODEL", "small")
    device: str = _env("STT_DEVICE", "auto")             # "cpu", "cuda" or "auto"
    compute_type: str = _env("STT_COMPUTE_TYPE", "int8")  # int8 on CPU, float16 on GPU
    threads: int = int(_env("STT_THREADS", "2"))

    sample_rate: int = 16_000                               # what Whisper models expect
    max_upload_mb: int = int(_env("MAX_UPLOAD_MB", "500"))
    max_duration_s: float = float(_env("MAX_DURATION_S", str(4 * 3600)))
    # Requests longer than this must use the async /jobs endpoint instead of /transcribe.
    sync_limit_s: float = float(_env("SYNC_LIMIT_S", "600"))
    # How many inferences may run at once (CPU-bound; 1 per CPU socket / GPU is sane).
    max_concurrency: int = int(_env("MAX_CONCURRENCY", "1"))

    # Silence detection used to choose chunk boundaries.
    silence_db: float = float(_env("SILENCE_DB", "-35"))
    min_silence_s: float = float(_env("MIN_SILENCE_S", "0.4"))


def get_settings() -> Settings:
    return Settings()
