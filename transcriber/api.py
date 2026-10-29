"""HTTP service.

POST /transcribe   short audio, synchronous; returns JSON (or SRT/VTT/TXT) with segment timestamps
POST /jobs         long audio, asynchronous; returns a job id immediately (202)
GET  /jobs/{id}    job status, progress (chunks done / total) and the result when finished
GET  /health       liveness + which engine/model is loaded

Run:  uvicorn transcriber.api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
import shutil
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import PlainTextResponse

from .audio import AudioError, probe
from .config import Settings, get_settings
from .engines import Engine, load_engine
from .formats import RENDERERS
from .pipeline import transcribe_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
Format = Literal["json", "srt", "vtt", "txt"]


class State:
    settings: Settings
    engine: Engine
    lock: threading.Semaphore
    jobs: dict[str, dict]
    executor: ThreadPoolExecutor


state = State()


def create_app(engine: Engine | None = None, settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        state.settings = settings or get_settings()
        state.engine = engine or load_engine(state.settings)   # load the model once
        state.lock = threading.Semaphore(state.settings.max_concurrency)
        state.jobs = {}
        state.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt-job")
        yield
        state.executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="Transcription service", version="0.1.0", lifespan=lifespan)

    async def save_upload(upload: UploadFile, folder: Path) -> Path:
        """Stream the upload to disk with a size cap (never hold it all in memory)."""
        limit = state.settings.max_upload_mb * 1024 * 1024
        dest = folder / ("upload" + Path(upload.filename or "").suffix.lower()[:10])
        size = 0
        with dest.open("wb") as f:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"File larger than {state.settings.max_upload_mb} MB")
                f.write(chunk)
        if size == 0:
            raise HTTPException(400, "Empty file")
        return dest

    def render(result, fmt: str):
        if fmt == "json":
            return result.as_dict()
        fn, media = RENDERERS[fmt]
        return PlainTextResponse(fn(result), media_type=media)

    @app.get("/health")
    def health():
        return {"status": "ok", "engine": state.engine.name, "model": state.engine.model_name}

    @app.post("/transcribe")
    async def transcribe(file: UploadFile = File(...),
                         language: str | None = Query(None, description="ISO code, e.g. 'en'; omit to auto-detect"),
                         format: Format = Query("json")):
        tmp = Path(tempfile.mkdtemp(prefix="stt-upload-"))
        try:
            path = await save_upload(file, tmp)
            try:
                info = await run_in_threadpool(probe, path)
            except AudioError as e:
                raise HTTPException(415, str(e))
            if info.duration > state.settings.sync_limit_s:
                raise HTTPException(413, f"Audio is {info.duration:.0f}s; use POST /jobs for files "
                                         f"longer than {state.settings.sync_limit_s:.0f}s")
            try:
                result = await run_in_threadpool(transcribe_file, path, state.engine, state.settings,
                                                 language, None, state.lock)
            except AudioError as e:
                raise HTTPException(415, str(e))
            return render(result, format)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @app.post("/jobs", status_code=202)
    async def create_job(file: UploadFile = File(...), language: str | None = Query(None)):
        tmp = Path(tempfile.mkdtemp(prefix="stt-job-"))
        try:
            path = await save_upload(file, tmp)
            info = await run_in_threadpool(probe, path)
        except AudioError as e:
            shutil.rmtree(tmp, ignore_errors=True)
            raise HTTPException(415, str(e))
        except HTTPException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        job_id = uuid.uuid4().hex
        job = {"id": job_id, "status": "queued", "progress": {"done": 0, "total": None},
               "duration": info.duration, "result": None, "error": None}
        state.jobs[job_id] = job

        def run():
            job["status"] = "running"
            try:
                def progress(done, total):
                    job["progress"] = {"done": done, "total": total}
                job["result"] = transcribe_file(path, state.engine, state.settings, language,
                                                progress, state.lock).as_dict()
                job["status"] = "done"
            except Exception as e:                  # surfaced to the client, not swallowed
                job["status"], job["error"] = "failed", str(e)[:500]
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

        state.executor.submit(run)
        return {"id": job_id, "status": "queued", "status_url": f"/jobs/{job_id}"}

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str, format: Format = Query("json")):
        job = state.jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Unknown job id")
        if format != "json" and job["status"] == "done":
            from .pipeline import Transcript
            from .postprocess import Segment
            r = job["result"]
            t = Transcript(r["text"], [Segment(**s) for s in r["segments"]], r["language"],
                           r["duration"], r["metadata"])
            return render(t, format)
        return job

    return app


app = create_app()
