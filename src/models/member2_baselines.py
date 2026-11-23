"""Evaluate the three remaining pretrained Member 2 ASR models."""

import argparse
import csv
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import soundfile
import torch

from src.evaluation.evaluate import evaluate_model, load_references


ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    "qwen3_asr_06b": (
        "Qwen/Qwen3-ASR-0.6B", "5eb144179a02acc5e5ba31e748d22b0cf3e303b0",
        "79d6cbd4c98c7bbffe9db2edac07f56cd6637d0d5944b27f6c2b8353840323ea",
    ),
    "canary_1b_v2": (
        "nvidia/canary-1b-v2", "d455706339a6b32e1aa40f82c713a482a0c938e2",
        "4f6d381f6a939b95e9f8516212148b701e9b3805d26462c4c3e9e34f3d89b6f8",
    ),
    "parakeet_tdt_06b_v3": (
        "nvidia/parakeet-tdt-0.6b-v3", "541d1f99c6b0c3cd0b11a95167540bb8edefd82b",
        "3a2026366188c8c68598edbbff92f8d11590a08e0ae2e6775544e7b07d6a5e11",
    ),
}
DECODING = {
    "qwen3_asr_06b": {"language": "Spanish", "max_new_tokens": 256},
    "canary_1b_v2": {
        "source_language": "es", "target_language": "es", "max_new_tokens": 128,
        "num_beams": 1, "do_sample": False,
    },
    "parakeet_tdt_06b_v3": {"language": "automatic", "strategy": "greedy TDT"},
}
COLUMNS = ("audio_id", "prediction", "inference_time_seconds")
SAMPLE_RATE = 16000


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_clips(path):
    with path.open(encoding="utf-8") as source:
        return [
            (row["audio_id"], ROOT / row["audio_path"], row["duration"])
            for row in map(json.loads, source)
        ]


def read_audio(path):
    audio, rate = soundfile.read(path, dtype="float32")
    if rate != SAMPLE_RATE or audio.ndim != 1 or not audio.size:
        raise ValueError(f"Expected a nonempty 16 kHz mono WAV: {path}")
    return audio


def build_decoder(key, source, device, dtype, batch_size):
    config = DECODING[key]
    if key == "qwen3_asr_06b":
        from qwen_asr import Qwen3ASRModel

        model = Qwen3ASRModel.from_pretrained(
            str(source), dtype=dtype, device_map=str(device),
            max_inference_batch_size=batch_size,
            max_new_tokens=config["max_new_tokens"],
            local_files_only=True,
        )

        def decode(waves):
            results = model.transcribe(
                [(wave, SAMPLE_RATE) for wave in waves], language=config["language"]
            )
            return [result.text for result in results]

        return decode

    from transformers import AutoModelForSpeechSeq2Seq, AutoModelForTDT, AutoProcessor

    processor = AutoProcessor.from_pretrained(source, local_files_only=True)
    canary = key == "canary_1b_v2"
    model_class = AutoModelForSpeechSeq2Seq if canary else AutoModelForTDT
    model = model_class.from_pretrained(
        source, dtype=dtype, device_map=str(device), use_safetensors=True, local_files_only=True,
    ).eval()

    if canary:
        def decode(waves):
            inputs = processor.apply_transcription_request(
                audio=waves, source_language=config["source_language"],
                target_language=config["target_language"],
            ).to(device, dtype=dtype)
            ids = model.generate(**inputs, max_new_tokens=config["max_new_tokens"],
                                 num_beams=config["num_beams"], do_sample=config["do_sample"])
            return processor.decode(ids, skip_special_tokens=True)
    else:
        def decode(waves):
            inputs = processor(waves, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(device, dtype=dtype)
            output = model.generate(**inputs, return_dict_in_generate=True)
            return processor.decode(output.sequences, skip_special_tokens=True)

    return decode


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def predict(decode, clips, device, batch_size):
    ordered = sorted(clips, key=lambda clip: clip[2])
    predictions = {}
    for i in range(0, len(ordered), batch_size):
        chunk = ordered[i:i + batch_size]
        waves = [read_audio(path) for _, path, _ in chunk]
        sync(device)
        start = perf_counter()
        hypotheses = decode(waves)
        sync(device)
        per_clip = (perf_counter() - start) / len(chunk)
        if isinstance(hypotheses, str) and len(chunk) == 1:
            hypotheses = [hypotheses]
        for (audio_id, _, _), hypothesis in zip(chunk, hypotheses, strict=True):
            if not isinstance(hypothesis, str):
                raise TypeError(f"Non-text hypothesis for {audio_id}")
            if audio_id in predictions:
                raise ValueError(f"Duplicate audio ID: {audio_id}")
            predictions[audio_id] = (audio_id, hypothesis, per_clip)
        if len(predictions) % 256 < batch_size or len(predictions) == len(ordered):
            print(f"Decoded {len(predictions)}/{len(ordered)}", flush=True)
    return [predictions[audio_id] for audio_id, _, _ in clips]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    temporary.replace(path)


def run(args):
    key = args.model
    model_id, revision, expected_hash = MODELS[key]
    source = args.work / "models" / key
    dev_path = args.work / "dev" / f"{key}_run_manifest.json"
    weights_hash = sha256(source / "model.safetensors")
    if weights_hash != expected_hash:
        raise ValueError(f"Weights do not match {model_id} at revision {revision}.")
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    runtime_versions = {
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "transformers": version("transformers"),
    }
    if key == "qwen3_asr_06b":
        runtime_versions["qwen_asr"] = version("qwen-asr")
    settings = {
        "model": model_id,
        "revision": revision,
        "checkpoint": "model.safetensors",
        "weights_sha256": weights_hash,
        "task": "Spanish transcription",
        "decoding": DECODING[key],
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "dtype": str(dtype),
        "batch_size": args.batch_size,
        "versions": runtime_versions,
        "inference_timing": "Synchronized batch time divided by size. Includes processing, generation and text decoding; excludes reading WAVs.",
    }
    split_path = ROOT / "data" / "splits" / f"{args.split}.jsonl"
    clips = read_clips(split_path)
    if args.split == "validation":
        ordered = sorted(clips, key=lambda clip: clip[2])
        clips = [ordered[round(i * (len(ordered) - 1) / 7)] for i in range(8)]
        csv_path = args.work / "dev" / f"{key}.csv"
        manifest_path = dev_path
    else:
        dev = json.loads(dev_path.read_text(encoding="utf-8"))
        if dev["settings"] != settings:
            raise ValueError("Model or runtime settings differ from the validation run.")
        csv_path = ROOT / "data" / "predictions" / f"{key}.csv"
        manifest_path = ROOT / "results" / "member2" / f"{key}_run_manifest.json"
        if csv_path.exists() or manifest_path.exists():
            raise FileExistsError(f"Test output already exists for {key}.")

    print(f"Loading {settings['model']} at {settings['revision']}", flush=True)
    decode = build_decoder(key, source, device, dtype, args.batch_size)
    with torch.inference_mode():
        silence = torch.zeros(SAMPLE_RATE).numpy()
        decode([silence])
    rows = predict(decode, clips, device, args.batch_size)
    if not any(text.strip() for _, text, _ in rows):
        raise RuntimeError(f"{key} produced only empty transcripts.")
    manifest = {
        "settings": settings,
        "split": args.split,
        "split_sha256": sha256(split_path),
        "utterances": len(rows),
        "validation_audio_ids": [row[0] for row in rows] if args.split == "validation" else dev["validation_audio_ids"],
    }
    write_csv(csv_path, rows)
    write_json(manifest_path, manifest)
    print(f"Predictions: {csv_path}\nRun Manifest: {manifest_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("split", choices=("validation", "test", "score"))
    parser.add_argument("--work", type=Path, default=ROOT.parent / ".task2-work")
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.split == "score":
        predictions = ROOT / "data" / "predictions" / f"{args.model}.csv"
        metrics = evaluate_model(predictions, load_references())
        write_json(ROOT / "results" / "member2" / f"{args.model}_metrics.json", metrics)
        print(f"{args.model}: normalized WER {metrics['normalized']['WER']:.2%}, "
              f"CER {metrics['normalized']['CER']:.2%}", flush=True)
    else:
        run(args)


if __name__ == "__main__":
    main()
