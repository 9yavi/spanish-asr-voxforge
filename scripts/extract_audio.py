
from datasets import load_dataset, Audio
import soundfile as sf
import io
import json
from pathlib import Path

DATASET_NAME = "ciempiess/voxforge_spanish"

AUDIO_DIR = Path("data/raw/audio")
METADATA_FILE = Path("data/raw/metadata.jsonl")

AUDIO_DIR.mkdir(parents=True, exist_ok=True)

print("Loading dataset...", flush=True)

dataset = load_dataset(
    DATASET_NAME,
    split="train"
)

dataset = dataset.cast_column(
    "audio",
    Audio(decode=False)
)

print(f"Total samples: {len(dataset)}", flush=True)
print("Starting extraction...", flush=True)

errors = 0

with open(METADATA_FILE, "w", encoding="utf-8") as metadata_file:

    for index, sample in enumerate(dataset):

        try:
            audio_info = sample["audio"]
            audio_bytes = audio_info["bytes"]

            if audio_bytes is None:
                raise ValueError("Audio bytes are missing")

            audio, sample_rate = sf.read(
                io.BytesIO(audio_bytes)
            )

            if sample_rate != 16000 or audio.ndim != 1:
                raise ValueError(
                    f"Unexpected audio format: "
                    f"{sample_rate} Hz, shape {audio.shape}"
                )

            audio_id = sample["audio_id"]
            output_path = AUDIO_DIR / f"{audio_id}.wav"

            sf.write(output_path, audio, sample_rate)

            metadata = {
                "audio_id": audio_id,
                "speaker_id": sample["speaker_id"],
                "country": sample["country"],
                "gender": sample["gender"],
                "duration": sample["duration"],
                "text": sample["normalized_text"],
                "audio_path": str(output_path),
                "sample_rate": sample_rate
            }

            metadata_file.write(
                json.dumps(metadata, ensure_ascii=False) + "\n"
            )

        except Exception as error:
            errors += 1
            print(
                f"Error at sample {index}: {error}",
                flush=True
            )

        if (index + 1) % 500 == 0:
            print(
                f"Processed {index + 1}/{len(dataset)} "
                f"| Errors: {errors}",
                flush=True
            )

print("\nExtraction finished.")
print(f"Total samples: {len(dataset)}")
print(f"Errors: {errors}")
print(f"Audio directory: {AUDIO_DIR}")
print(f"Metadata file: {METADATA_FILE}")
