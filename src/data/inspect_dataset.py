
import json
import soundfile as sf
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
AUDIO_DIR = ROOT / "data" / "raw" / "audio"
METADATA_FILE = ROOT / "data" / "raw" / "metadata.jsonl"

print("=== LOCAL DATASET INSPECTION ===")

if not METADATA_FILE.exists():
    raise FileNotFoundError(f"Metadata not found: {METADATA_FILE}")

if not AUDIO_DIR.exists():
    raise FileNotFoundError(f"Audio directory not found: {AUDIO_DIR}")

records = []

with METADATA_FILE.open("r", encoding="utf-8") as file:
    for line_number, line in enumerate(file, 1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"Invalid JSON at line {line_number}")

print(f"\nMetadata records: {len(records)}")

missing = []
corrupt = []
invalid_audio = []
missing_text = []
speaker_counts = Counter()
total_duration = 0.0

for record in records:
    audio_id = record.get("audio_id")
    speaker_id = record.get("speaker_id")
    transcript = record.get("text")

    if speaker_id:
        speaker_counts[speaker_id] += 1

    if not transcript or not transcript.strip():
        missing_text.append(audio_id)

    if not audio_id:
        missing.append("Missing audio_id")
        continue

    audio_path = AUDIO_DIR / f"{audio_id}.wav"

    if not audio_path.exists():
        missing.append(audio_id)
        continue

    try:
        info = sf.info(str(audio_path))
        if info.samplerate != 16000 or info.channels != 1:
            invalid_audio.append(
                (audio_id, info.samplerate, info.channels)
            )
        total_duration += info.duration
    except Exception as error:
        corrupt.append((audio_id, str(error)))

print("\n=== RESULTS ===")
print(f"Unique speakers: {len(speaker_counts)}")
print(f"Missing audio: {len(missing)}")
print(f"Corrupt audio: {len(corrupt)}")
print(f"Invalid sample rate/channels: {len(invalid_audio)}")
print(f"Empty transcripts: {len(missing_text)}")
print(f"Total audio duration: {total_duration / 3600:.2f} hours")

print("\n=== DETAILS ===")
print("Missing:", missing[:10])
print("Corrupt:", corrupt[:10])
print("Invalid audio:", invalid_audio[:10])
print("Empty transcripts:", missing_text[:10])

print("\n=== INSPECTION COMPLETED ===")
