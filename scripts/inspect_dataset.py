
from datasets import load_dataset, Audio

DATASET_NAME = "ciempiess/voxforge_spanish"

print("Connecting to Hugging Face...", flush=True)

dataset = load_dataset(
    DATASET_NAME,
    split="train",
    streaming=True
)

dataset = dataset.cast_column("audio", Audio(decode=False))

print("\n=== DATASET FEATURES ===")

for column, feature in dataset.features.items():
    print(f"{column}: {feature}")

print("\n=== FIRST SAMPLE ===")

sample = next(iter(dataset))

for key, value in sample.items():
    if key == "audio":
        print(f"{key}: Audio metadata available")
        print(f"Audio keys: {list(value.keys())}")
    else:
        print(f"{key}: {value}")

print("\n=== INSPECTION COMPLETED ===")
