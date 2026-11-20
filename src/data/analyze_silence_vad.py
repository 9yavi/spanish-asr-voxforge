
import json
import csv
import numpy as np
import soundfile as sf
import webrtcvad
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METADATA = ROOT / "data/raw/metadata.jsonl"
OUTPUT = ROOT / "data/splits/silence_analysis_vad.csv"

SAMPLE_RATE = 16000
FRAME_MS = 20
VAD_MODE = 2

with METADATA.open(encoding="utf-8") as f:
    records = [json.loads(line) for line in f]

results = []
total_duration = 0
total_non_speech = 0
clips_with_non_speech = 0

frame_size = int(SAMPLE_RATE * FRAME_MS / 1000)

for index, record in enumerate(records, 1):
    audio, sample_rate = sf.read(record["audio_path"])

    if sample_rate != SAMPLE_RATE or audio.ndim != 1:
        raise ValueError(f"Invalid audio format: {record['audio_id']}")

    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)

    usable_length = len(audio_int16) // frame_size * frame_size
    frames = audio_int16[:usable_length].reshape(-1, frame_size)

    # Create a fresh VAD instance for each recording
    vad = webrtcvad.Vad(VAD_MODE)

    non_speech_frames = 0

    for frame in frames:
        is_speech = vad.is_speech(frame.tobytes(), SAMPLE_RATE)

        if not is_speech:
            non_speech_frames += 1

    analyzed_duration = len(frames) * FRAME_MS / 1000
    non_speech_duration = non_speech_frames * FRAME_MS / 1000

    percentage = (
        non_speech_duration / analyzed_duration * 100
        if analyzed_duration > 0 else 0
    )

    total_duration += analyzed_duration
    total_non_speech += non_speech_duration

    if non_speech_duration > 0:
        clips_with_non_speech += 1

    results.append({
        "audio_id": record["audio_id"],
        "duration_seconds": round(analyzed_duration, 3),
        "non_speech_seconds": round(non_speech_duration, 3),
        "non_speech_percentage": round(percentage, 2)
    })

    if index % 1000 == 0:
        print(f"Processed {index}/{len(records)} recordings")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\nVAD Analysis Complete")
print(f"VAD Mode: {VAD_MODE}")
print(f"Frame duration: {FRAME_MS} ms")
print(f"Total clips: {len(records)}")
print(f"Clips with non-speech: {clips_with_non_speech}")
print(f"Total analyzed duration: {total_duration / 3600:.2f} hours")
print(f"Estimated non-speech duration: {total_non_speech / 3600:.2f} hours")

if total_duration > 0:
    print(
        f"Non-speech percentage: "
        f"{total_non_speech / total_duration * 100:.2f}%"
    )

print(f"Results saved to: {OUTPUT}")
