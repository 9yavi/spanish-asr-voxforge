
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data/raw/metadata.jsonl"
OUTPUT_DIR = ROOT / "data/splits"

SEED = 42

RATIOS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15
}

random.seed(SEED)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading metadata...", flush=True)

# Group samples by speaker
groups = defaultdict(list)

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    for line in file:
        sample = json.loads(line)
        groups[sample["speaker_id"]].append(sample)

speakers = list(groups.items())
random.shuffle(speakers)

# Process larger speaker groups first
speakers.sort(
    key=lambda item: len(item[1]),
    reverse=True
)

splits = {
    "train": [],
    "validation": [],
    "test": []
}

total_samples = sum(len(samples) for samples in groups.values())

targets = {
    name: total_samples * ratio
    for name, ratio in RATIOS.items()
}

# Assign each speaker to exactly one split
for speaker_id, samples in speakers:

    split_name = max(
        RATIOS,
        key=lambda name: (
            targets[name] - len(splits[name])
        ) / targets[name]
    )

    splits[split_name].extend(samples)

print("\nSaving splits...\n")

# Save files and calculate statistics
for split_name, samples in splits.items():

    output_file = OUTPUT_DIR / f"{split_name}.jsonl"

    with open(output_file, "w", encoding="utf-8") as file:
        for sample in samples:
            file.write(
                json.dumps(sample, ensure_ascii=False) + "\n"
            )

    speaker_count = len({
        sample["speaker_id"] for sample in samples
    })

    duration_hours = sum(
        sample["duration"] for sample in samples
    ) / 3600

    percentage = len(samples) / total_samples * 100

    print(f"{split_name.upper()}")
    print(f"Samples: {len(samples)}")
    print(f"Speakers: {speaker_count}")
    print(f"Duration: {duration_hours:.2f} hours")
    print(f"Percentage: {percentage:.2f}%")
    print("-" * 30)

# Verify no speaker overlap
speaker_sets = {
    name: {
        sample["speaker_id"] for sample in samples
    }
    for name, samples in splits.items()
}

assert speaker_sets["train"].isdisjoint(
    speaker_sets["validation"]
)

assert speaker_sets["train"].isdisjoint(
    speaker_sets["test"]
)

assert speaker_sets["validation"].isdisjoint(
    speaker_sets["test"]
)

# Verify all samples were assigned
assert sum(
    len(samples) for samples in splits.values()
) == total_samples

print("\nSplit completed successfully.")
print("Total samples:", total_samples)
print("No speaker overlap detected.")
print("All samples accounted for.")
