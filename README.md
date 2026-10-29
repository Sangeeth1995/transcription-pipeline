# Transcription pipeline

A small, production-minded service that turns an audio (or video) file into text with
per-segment timestamps, ready for downstream use (JSON for systems, SRT/VTT for players).

```
upload ─► probe (ffprobe) ─► normalise (ffmpeg → 16 kHz mono PCM WAV)
       ─► find silences ─► plan chunks (cut in pauses, skip silence)
       ─► transcribe each chunk (Whisper) ─► shift to absolute time
       ─► post-process (clean, de-duplicate, flag) ─► JSON / SRT / VTT / TXT
```

## Run it

```bash
pip install -r requirements.txt          # needs ffmpeg on PATH (brew/apt install ffmpeg)

# CLI
python -m transcriber.cli meeting.mp3 --format srt -o meeting.srt

# Service
uvicorn transcriber.api:app --port 8000
curl -F file=@meeting.mp3 "localhost:8000/transcribe?format=json"
curl -F file=@podcast_2h.m4a localhost:8000/jobs          # long audio → job id (202)
curl localhost:8000/jobs/<id>                             # status, progress, result

# Docker
docker build -t transcriber . && docker run -p 8000:8000 -v hf-cache:/root/.cache transcriber
```

Configuration is by environment variable (`transcriber/config.py`): `STT_ENGINE`
(`faster-whisper` default, or `sherpa-onnx`), `STT_MODEL` (`tiny`…`large-v3`, or a model
directory), `STT_DEVICE`, `STT_COMPUTE_TYPE`, `SYNC_LIMIT_S`, `MAX_UPLOAD_MB`, `MAX_DURATION_S`.

## API

| Endpoint | Purpose |
| --- | --- |
| `POST /transcribe?language=en&format=json\|srt\|vtt\|txt` | Short files (≤ `SYNC_LIMIT_S`, default 10 min), synchronous |
| `POST /jobs` → `202 {id, status_url}` | Long files, processed in the background |
| `GET /jobs/{id}?format=…` | `queued / running / done / failed`, progress `{done, total}` chunks, result |
| `GET /health` | Liveness and loaded engine/model |

Response shape (illustrative faster-whisper values; real sherpa-onnx outputs are in `examples/` —
that English-only ONNX model reports no confidence or language, so those fields are `null` there):

```json
{
  "text": "God, as a direct consequence of the sin ...",
  "language": "en",
  "duration": 16.715,
  "segments": [
    {"id": 0, "start": 0.0, "end": 6.0, "text": "God, as a direct consequence ...", "confidence": 0.91, "flags": []}
  ],
  "metadata": {"engine": "faster-whisper", "model": "small",
               "source": {"format": "mp3", "codec": "mp3", "sample_rate": 44100, "channels": 2, "duration": 16.74},
               "chunks": 1, "speech_seconds": 16.3, "failed_chunks": [], "processing_seconds": 2.1, "real_time_factor": 0.13}
}
```

Errors: `400` empty file, `413` too large / too long for the sync endpoint (message points to `/jobs`),
`415` not decodable audio.

## Key engineering decisions

**Engine.** Whisper via `faster-whisper` (CTranslate2): open source, multilingual, robust to
noise and accents, returns segment timestamps, ~4× faster than the reference implementation, int8
on CPU and float16 on GPU. Engines sit behind a three-method interface (`engines.py`), so a hosted
API (Deepgram, AssemblyAI, Azure) or another model is a new class, not a rewrite. A second engine,
`sherpa-onnx` (Whisper exported to ONNX), runs fully offline and is what the tests in this repo
were verified with.

**Audio formats.** Every input goes through one ffmpeg normalisation step to 16 kHz, mono,
16-bit PCM WAV (Whisper's native input). The format is detected from the file's content with
ffprobe, not trusted from the extension; video files work (first audio track is used); 8 kHz
phone audio, 44.1/48 kHz stereo music, AAC, Opus, FLAC, etc. all reduce to the same thing. Anything
ffmpeg cannot decode is rejected up front with HTTP 415. Uploads are streamed to disk with a size cap.

**Long audio.**
1. *Two modes:* short files are synchronous; long files go to an async job with progress polling,
   so no HTTP request hangs for minutes.
2. *Chunk at pauses:* ffmpeg `silencedetect` (streaming, constant memory) finds silences; chunks are
   cut inside them so no word is split, with 150 ms padding. Speech with no pause is force-cut at
   the quietest 50 ms frame near the limit.
3. *Skip silence:* silent stretches are never sent to the model — less compute, and no Whisper
   "hallucinations" on silence. An all-silent file returns an empty transcript.
4. *Bounded memory:* each chunk is read from the WAV on disk by offset; a 4-hour file never sits in RAM.
5. *Absolute timestamps:* chunk-relative times are shifted by the chunk start.
6. *Fault isolation:* each chunk is retried once; a chunk that still fails is reported in
   `failed_chunks` with its time range, instead of losing the whole file.
7. *Ready to scale:* chunks are independent, so they can be fanned out to a worker pool/GPUs.
   Production would swap the in-memory job store for a queue (Celery/RQ + Redis) and object storage.

**Post-processing.** Whitespace clean-up, monotonic and clamped timestamps, merge of Whisper's
repetition loops (`repeated` flag), `low_confidence` flag from the decoder's average log-probability,
removal of segments the model marks as non-speech with low confidence. Output as JSON, SRT, VTT, TXT.

## Tested

`pytest -q` → 20 tests (formats, content-based detection, silence detection, chunk planning, forced
cuts, absolute timestamps, retries, post-processing, subtitle formats, the HTTP API and job flow, a
mocked faster-whisper adapter, and an end-to-end real-model test when `SHERPA_MODEL_DIR` is set).

Real-model runs (Whisper tiny.en, ONNX, 2 CPU cores), inputs built by `scripts/make_samples.sh`:

| Input | Result |
| --- | --- |
| 16 kHz WAV, 8 kHz WAV, 44.1 kHz stereo MP3, AAC (.m4a), Opus (.ogg), MP4 video | Correct text with segment timestamps |
| 10 s silence | Empty transcript, no hallucination |
| 4.2 min MP3 | 17 chunks, 40 segments, timestamps match the clip boundaries; 26 s (0.10 × real time) |
| Text file renamed `.mp3` | HTTP 415 |

## Next steps

Speaker diarization (pyannote), word-level timestamps (`word_timestamps=True`), per-chunk language
detection for code-switched audio, GPU batching, streaming/real-time input over WebSocket, PII
redaction before downstream storage, and WER tracking on a labelled evaluation set per language.
