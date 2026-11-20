from datasets import load_dataset, Audio
from collections import Counter

DATASET_NAME = "ciempiess/voxforge_spanish"

print("Downloading dataset...", flush=True)

dataset = load_dataset(
    DATASET_NAME,
    split="train"
)

print("Dataset downloaded successfully.", flush=True)

dataset = dataset.cast_column(
    "audio",
    Audio(decode=False)
)

print("Analyzing dataset...", flush=True)

total_samples = len(dataset)
total_duration = sum(dataset["duration"])
speakers = set(dataset["speaker_id"])

countries = Counter(dataset["country"])
genders = Counter(dataset["gender"])

print("\n========== DATASET REPORT ==========")
print(f"Total samples: {total_samples}")
print(f"Unique speakers: {len(speakers)}")
print(f"Total duration: {total_duration / 3600:.2f} hours")

print("\nSamples by country:")
for country, count in countries.most_common():
    print(f"{country}: {count}")

print("\nSamples by gender:")
for gender, count in genders.most_common():
    print(f"{gender}: {count}")

print("\n========== ANALYSIS COMPLETED ==========")