import json
import soundfile as sf
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw"
CLEANED = ROOT / "data/cleaned"

METADATA = RAW / "metadata.jsonl"
OUTPUT = CLEANED / "metadata.jsonl"
REPORT = CLEANED / "cleaning_report.json"

CLEANED.mkdir(parents=True, exist_ok=True)

accepted = []
excluded = []
text_changes = 0

with METADATA.open(encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

for record in records:
    audio_id = record.get("audio_id")
    text = record.get("text", "")
    audio_path = RAW / "audio" / f"{audio_id}.wav"

    if not audio_id:
        excluded.append({"audio_id": None, "reason": "missing_audio_id"})
        continue

    if not isinstance(text, str) or not text.strip():
        excluded.append({"audio_id": audio_id, "reason": "empty_transcript"})
        continue

    if not audio_path.is_file():
        excluded.append({"audio_id": audio_id, "reason": "missing_audio"})
        continue

    try:
        info = sf.info(str(audio_path))
        if info.samplerate != 16000 or info.channels != 1:
            excluded.append({"audio_id": audio_id, "reason": "invalid_audio_format"})
            continue
        if info.frames == 0:
            excluded.append({"audio_id": audio_id, "reason": "empty_audio"})
            continue
    except Exception as e:
        excluded.append({
            "audio_id": audio_id,
            "reason": f"unreadable_audio: {str(e)}"
        })
        continue

    cleaned_text = unicodedata.normalize("NFC", text)
    cleaned_text = " ".join(cleaned_text.split())

    if cleaned_text != text:
        text_changes += 1

    cleaned_record = dict(record)
    cleaned_record["text"] = cleaned_text
    accepted.append(cleaned_record)

with OUTPUT.open("w", encoding="utf-8") as f:
    for record in accepted:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

report = {
    "input_records": len(records),
    "accepted_records": len(accepted),
    "excluded_records": len(excluded),
    "transcripts_normalized": text_changes,
    "audio_validation": "16kHz mono, readable, non-empty",
    "audio_files_modified": 0,
    "exclusions": excluded
}

with REPORT.open("w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(json.dumps(report, ensure_ascii=False, indent=2))
print(f"\nCleaned metadata: {OUTPUT}")
print(f"Cleaning report: {REPORT}")
