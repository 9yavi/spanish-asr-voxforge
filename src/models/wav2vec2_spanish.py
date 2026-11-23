"""Run Spanish wav2vec2 CTC greedy inference on project manifests."""
import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import soundfile as sf
import torch
import transformers
from transformers import AutoModelForCTC, Wav2Vec2Processor

ROOT = Path(__file__).resolve().parents[2]
MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"
REVISION = "96d7e9b4e4a78af515a3c6d3cee7c0826045d276"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.split == "test" and args.limit:
        parser.error("Test must cover the complete split.")
    if args.output.exists():
        raise FileExistsError(args.output)
    path = args.data_root / "data" / "splits" / (args.split + ".jsonl")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if args.limit:
        records = records[:args.limit]
    processor = Wav2Vec2Processor.from_pretrained(args.model_dir, local_files_only=True)
    device = torch.device("cuda")
    model = AutoModelForCTC.from_pretrained(args.model_dir, local_files_only=True,
                                           dtype=torch.bfloat16).to(device).eval()
    predictions = []
    started = time.perf_counter()
    total_duration = 0.0
    with torch.inference_mode():
        for index, row in enumerate(records, 1):
            audio, rate = sf.read(args.data_root / row["audio_path"], dtype="float32")
            if rate != 16000 or audio.ndim != 1:
                raise ValueError(row["audio_id"])
            inputs = processor(audio, sampling_rate=rate, return_tensors="pt")
            inputs = {key: value.to(device, dtype=torch.bfloat16 if value.is_floating_point() else value.dtype)
                      for key, value in inputs.items()}
            torch.cuda.synchronize()
            start = time.perf_counter()
            ids = model(**inputs).logits.argmax(dim=-1)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            predictions.append((row["audio_id"], processor.batch_decode(ids)[0], elapsed))
            total_duration += len(audio) / rate
            if index % 128 == 0 or index == len(records):
                print(f"Processed {index}/{len(records)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("audio_id", "prediction", "inference_time_seconds"))
        writer.writerows(predictions)
    info = {
        "model": MODEL, "revision": REVISION, "split": args.split,
        "split_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "utterances": len(records), "audio_seconds": total_duration,
        "wall_seconds": time.perf_counter() - started, "language": "es",
        "decoding": "CTC greedy", "batch_size": 1, "dtype": "bfloat16",
        "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
        "cuda": torch.version.cuda, "transformers": transformers.__version__,
        "python": platform.python_version(),
        "timing": "synchronized model forward pass per utterance",
    }
    manifest = args.output.with_suffix(".manifest.json")
    manifest.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info, indent=2), flush=True)


if __name__ == "__main__":
    main()
