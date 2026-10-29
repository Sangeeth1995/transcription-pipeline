FROM python:3.12-slim
# ffmpeg/ffprobe decode every input format; nothing else is needed at the OS level
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY transcriber ./transcriber
ENV STT_ENGINE=faster-whisper STT_MODEL=small STT_DEVICE=auto STT_COMPUTE_TYPE=int8
EXPOSE 8000
# model weights are cached under /root/.cache - mount a volume there to avoid re-downloading
HEALTHCHECK --interval=30s --start-period=120s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "transcriber.api:app", "--host", "0.0.0.0", "--port", "8000"]
