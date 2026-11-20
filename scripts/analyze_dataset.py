
import json
import hashlib
import re
import unicodedata
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "data/raw/metadata.jsonl"
AUDIO_DIR = ROOT / "data/raw/audio"

def normalize_text(text):
    text = unicodedata.normalize("NFC", text or "").lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())

def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

records = []

with METADATA.open(encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

countries = Counter(r.get("country", "unknown") for r in records)
genders = Counter(r.get("gender", "unknown") for r in records)
speakers = {r.get("speaker_id") for r in records if r.get("speaker_id")}

country_texts = defaultdict(list)
normalized_texts = defaultdict(list)

print("Analyzing local dataset...", flush=True)

audio_hashes = defaultdict(list)
missing_audio = []

for index, record in enumerate(records, 1):
    audio_id = record.get("audio_id")
    path = AUDIO_DIR / f"{audio_id}.wav" if audio_id else None

    if not path or not path.exists():
        missing_audio.append(audio_id)
    else:
        audio_hashes[file_hash(path)].append(audio_id)

    text = record.get("text", "")
    normalized = normalize_text(text)
    normalized_texts[normalized].append(audio_id)

    if index % 2000 == 0:
        print(f"Processed {index}/{len(records)} records", flush=True)

duplicate_audio = {
    h: ids for h, ids in audio_hashes.items() if len(ids) > 1
}

duplicate_texts = {
    text: ids for text, ids in normalized_texts.items()
    if text and len(ids) > 1
}

durations = [float(r.get("duration", 0)) for r in records]

print("\n========== DATASET REPORT ==========")
print(f"Total samples: {len(records)}")
print(f"Unique speakers: {len(speakers)}")
print(f"Total duration: {sum(durations) / 3600:.2f} hours")

print("\nSamples by country:")
for country, count in countries.most_common():
    print(f"{country}: {count}")

print("\nSamples by gender:")
for gender, count in genders.most_common():
    print(f"{gender}: {count}")

print("\n========== QUALITY CHECK ==========")
print(f"Missing audio: {len(missing_audio)}")
print(f"Duplicate audio groups: {len(duplicate_audio)}")
print(f"Audio files involved in duplicates: {sum(map(len, duplicate_audio.values()))}")
print(f"Duplicate normalized text groups: {len(duplicate_texts)}")
print(f"Records involved in duplicate texts: {sum(map(len, duplicate_texts.values()))}")

print("\nExamples of duplicate audio:")
for ids in list(duplicate_audio.values())[:10]:
    print(ids)

print("\nExamples of duplicate texts:")
for text, ids in list(duplicate_texts.items())[:10]:
    print(f"{text[:100]!r}: {ids}")

print("\n========== ANALYSIS COMPLETED ==========")
