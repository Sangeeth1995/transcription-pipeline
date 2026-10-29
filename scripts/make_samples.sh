#!/usr/bin/env bash
# Builds test inputs in several formats from three short LibriSpeech clips
# (shipped with the sherpa-onnx Whisper model): usage  scripts/make_samples.sh <test_wavs dir>
set -euo pipefail
SRC=${1:?path to test_wavs}; OUT=samples; mkdir -p "$OUT"
q="-nostdin -hide_banner -loglevel error -y"
cp "$SRC/0.wav" "$OUT/clip_16k.wav"
cp "$SRC/8k.wav" "$OUT/phone_8k.wav"                                            # 8 kHz telephone-rate WAV
ffmpeg $q -i "$SRC/1.wav" -ar 44100 -ac 2 -b:a 128k "$OUT/stereo_44k.mp3"     # stereo 44.1 kHz MP3
ffmpeg $q -i "$SRC/1.wav" -ar 48000 -c:a aac "$OUT/speech.m4a"                # AAC in MP4 container
ffmpeg $q -i "$SRC/0.wav" -c:a libopus "$OUT/speech.ogg"                      # Opus (browser / WhatsApp style)
ffmpeg $q -f lavfi -i color=c=blue:s=320x240:d=7 -i "$SRC/0.wav" -shortest -c:v libx264 -c:a aac "$OUT/video.mp4"
ffmpeg $q -f lavfi -i anullsrc=r=16000:cl=mono -t 10 "$OUT/silence.wav"       # should give no text
# ~3.5 minute "long" file: the three clips repeated, separated by pauses of varying length
ffmpeg $q -f lavfi -i anullsrc=r=16000:cl=mono -t 1.2 /tmp/gap.wav
ffmpeg $q -i "$SRC/8k.wav" -ar 16000 /tmp/c2.wav
: > /tmp/list.txt
for i in $(seq 1 8); do
  for c in "$SRC/0.wav" /tmp/gap.wav "$SRC/1.wav" /tmp/gap.wav /tmp/c2.wav /tmp/gap.wav; do echo "file '$c'" >> /tmp/list.txt; done
done
ffmpeg $q -f concat -safe 0 -i /tmp/list.txt -c:a pcm_s16le "$OUT/long_3min.wav"
ffmpeg $q -i "$OUT/long_3min.wav" -b:a 64k "$OUT/long_3min.mp3"
echo "not audio" > "$OUT/not_audio.txt"
ls -la "$OUT"
