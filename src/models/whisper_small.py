"""Run the Spanish Whisper-small baseline and train a decoder LoRA adapter."""

import argparse
import csv
import hashlib
import json
import platform
import random
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import jiwer
import soundfile as sf
import torch
import transformers
from peft import LoraConfig, __version__ as peft_version
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.evaluation.normalize_text import normalize_text


ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data" / "splits"
MODEL_ID = "openai/whisper-small"
REVISION = "973afd24965f72e36ca33b3055d56a652f456b4d"
BASE_SHA256 = "1d7734884874f1a1513ed9aa760a4f8e97aaa02fd6d93a3a85d27b2ae9ca596b"
SAMPLE_RATE = 16000
MAX_SECONDS = 30
SEED = 42
LANGUAGE = "es"
TASK = "transcribe"
MAX_NEW_TOKENS = 220
COLUMNS = ("audio_id", "prediction", "inference_time_seconds")


@dataclass(frozen=True)
class Clip:
    audio_id: str
    path: Path
    duration: float
    text: str | None


def read_split(split, *, with_text=False, limit=None):
    with (SPLITS / f"{split}.jsonl").open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    clips = [
        Clip(row["audio_id"], ROOT / row["audio_path"], row["duration"],
             row["text"] if with_text else None)
        for row in rows
    ]
    if not clips or len({clip.audio_id for clip in clips}) != len(clips):
        raise ValueError(f"Empty split or duplicate audio IDs: {split}")
    return clips[:limit]


def read_audio(clip):
    try:
        samples, rate = sf.read(clip.path, dtype="float32")
    except (OSError, RuntimeError) as error:
        raise RuntimeError(f"Cannot read {clip.audio_id}. Check the Git LFS audio download.") from error
    if rate != SAMPLE_RATE or samples.ndim != 1 or not 0 < len(samples) <= MAX_SECONDS * rate:
        raise ValueError(f"Expected nonempty mono 16 kHz audio up to 30 seconds: {clip.audio_id}")
    return samples


def model_setup(args, *, training=False, adapter=None):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install a CUDA-enabled PyTorch build.")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    source = str(args.base_dir) if args.base_dir else MODEL_ID
    processor = WhisperProcessor.from_pretrained(
        source, revision=REVISION, language=LANGUAGE, task=TASK
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        source, revision=REVISION, dtype=dtype, use_safetensors=True,
        attn_implementation="sdpa",
    ).to(device)
    if training:
        model.add_adapter(LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
            target_modules=r"model\.decoder\.layers\.\d+\.(self_attn|encoder_attn)\.(q_proj|v_proj)",
            base_model_name_or_path=MODEL_ID, revision=REVISION,
        ))
        # AdamW updates adapter weights in FP32. The frozen base uses the compute dtype.
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
    elif adapter is not None:
        model.load_adapter(str(adapter))
    return processor, model, device, dtype


def encode_audio(processor, clips, device, dtype):
    features = processor.feature_extractor(
        [read_audio(clip) for clip in clips], sampling_rate=SAMPLE_RATE,
        return_attention_mask=True, return_tensors="pt", device=str(device),
    )
    return {
        "input_features": features.input_features.to(device=device, dtype=dtype),
        "attention_mask": features.attention_mask.to(device),
    }


def batches(clips, batch_size):
    for start in range(0, len(clips), batch_size):
        yield clips[start:start + batch_size]


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def decode(model, processor, clips, device, dtype, batch_size, beams):
    model.eval()
    predictions = []
    ordered = sorted(clips, key=lambda clip: clip.duration)
    for batch in batches(ordered, batch_size):
        features = encode_audio(processor, batch, device, dtype)
        synchronize(device)
        started = perf_counter()
        ids = model.generate(
            **features, language=LANGUAGE, task=TASK, num_beams=beams,
            do_sample=False, max_new_tokens=MAX_NEW_TOKENS, use_cache=True,
        )
        synchronize(device)
        seconds = (perf_counter() - started) / len(batch)
        texts = processor.batch_decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        predictions.extend((clip.audio_id, text, seconds) for clip, text in zip(batch, texts, strict=True))
        if len(predictions) % 256 < batch_size or len(predictions) == len(clips):
            print(f"Decoded {len(predictions)}/{len(clips)} (beam={beams})", flush=True)
    by_id = {row[0]: row for row in predictions}
    return [by_id[clip.audio_id] for clip in clips]


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def manifest(split, device, dtype, batch_size, beams, clips, adapter):
    return {
        "model": MODEL_ID,
        "revision": REVISION,
        "base_weights_sha256": BASE_SHA256,
        "checkpoint": adapter.name if adapter else "pretrained",
        "adapter_sha256": file_hash(adapter / "adapter_model.safetensors") if adapter else None,
        "language": LANGUAGE,
        "task": TASK,
        "split": split,
        "split_sha256": file_hash(SPLITS / f"{split}.jsonl"),
        "utterances": len(clips),
        "preprocessing": "16 kHz mono WAV; Whisper log-Mel features",
        "decoding": {"num_beams": beams, "do_sample": False, "max_new_tokens": MAX_NEW_TOKENS},
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "device": str(device),
        "dtype": str(dtype),
        "batch_size": batch_size,
        "inference_timing": "Synchronized generate() batch wall time divided by batch size; excludes audio and decoding to text.",
        "versions": {
            "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "transformers": transformers.__version__, "peft": peft_version,
        },
    }


def write_predictions(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    tmp.replace(path)


def validation_subset(size):
    clips = read_split("validation", with_text=True)
    return random.Random(SEED).sample(clips, size) if size and size < len(clips) else clips


def dev_wer(clips, predictions):
    references = [normalize_text(clip.text) for clip in clips]
    return jiwer.wer(references, [normalize_text(row[1]) for row in predictions])


def train(args):
    args.work.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    train_clips = read_split("train", with_text=True)
    skipped = [clip.audio_id for clip in train_clips if clip.duration > MAX_SECONDS]
    train_clips = [clip for clip in train_clips if clip.duration <= MAX_SECONDS]
    eligible_train_count = len(train_clips)
    if args.train_size:
        train_clips = random.Random(SEED).sample(train_clips, args.train_size)
    dev_clips = validation_subset(args.dev_size)
    processor, model, device, dtype = model_setup(args, training=True)
    updates = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(updates, lr=args.learning_rate)
    history = []
    print(f"Train {len(train_clips)}, Dev {len(dev_clips)}, "
          f"trainable parameters {sum(parameter.numel() for parameter in updates):,}", flush=True)

    for epoch in range(1, args.epochs + 1):
        checkpoint = args.work / f"epoch-{epoch}"
        if checkpoint.exists():
            raise FileExistsError(f"Training checkpoint already exists: {checkpoint}")
        model.train()
        model.get_encoder().eval()
        random.shuffle(train_clips)
        started = perf_counter()
        loss_sum = torch.zeros((), device=device)
        for index, batch in enumerate(batches(train_clips, args.batch_size), 1):
            features = encode_audio(processor, batch, device, dtype)
            targets = processor.tokenizer([clip.text for clip in batch], padding=True, return_tensors="pt")
            labels = targets.input_ids[:, 1:].masked_fill(targets.attention_mask[:, 1:].ne(1), -100)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype == torch.bfloat16):
                loss = model(**features, labels=labels.to(device), use_cache=False).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(updates, max_norm=1.0)
            optimizer.step()
            loss_sum += loss.detach()
            if index == 1 or index % 25 == 0:
                print(f"Epoch {epoch}, batch {index}, mean loss {loss_sum.item() / index:.4f}, "
                      f"{perf_counter() - started:.1f}s", flush=True)
        mean_loss = loss_sum.item() / index
        if not torch.isfinite(loss_sum).item():
            raise RuntimeError(f"Non-finite training loss in epoch {epoch}")
        training_seconds = perf_counter() - started
        predictions = decode(model, processor, dev_clips, device, dtype, args.batch_size, 1)
        score = dev_wer(dev_clips, predictions)
        model.save_pretrained(checkpoint)
        config_path = checkpoint / "adapter_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["base_model_name_or_path"] = MODEL_ID
        write_json(config_path, config)
        history.append({"epoch": epoch, "checkpoint": checkpoint.name, "dev_wer": score,
                        "mean_training_loss": mean_loss, "training_seconds": training_seconds})
        best = min(history, key=lambda entry: entry["dev_wer"])
        print(f"Epoch {epoch}: validation WER {score:.2%}", flush=True)
        write_json(args.work / "training.json", {
            "base_model": MODEL_ID, "revision": REVISION, "method": "decoder attention LoRA",
            "seed": SEED, "train_count": len(train_clips), "eligible_train_count": eligible_train_count,
            "excluded_over_30_seconds": skipped, "dev_audio_ids": [clip.audio_id for clip in dev_clips],
            "train_sha256": file_hash(SPLITS / "train.jsonl"), "dev_sha256": file_hash(SPLITS / "validation.jsonl"),
            "batch_size": args.batch_size, "learning_rate": args.learning_rate, "optimizer": "AdamW",
            "history": history, "best_checkpoint": best["checkpoint"],
        })


def select(args):
    training = json.loads((args.work / "training.json").read_text(encoding="utf-8"))
    if training["dev_sha256"] != file_hash(SPLITS / "validation.jsonl"):
        raise ValueError("Validation manifest changed since training.")
    dev = {clip.audio_id: clip for clip in read_split("validation", with_text=True)}
    clips = [dev[audio_id] for audio_id in training["dev_audio_ids"]]
    adapter = args.work / training["best_checkpoint"]
    variants = {}
    for variant, checkpoint in (("baseline", None), ("finetuned", adapter)):
        processor, model, device, dtype = model_setup(args, adapter=checkpoint)
        trials = []
        for beams in (1, 3):
            predictions = decode(model, processor, clips, device, dtype, args.batch_size, beams)
            trials.append({"num_beams": beams, "dev_wer": dev_wer(clips, predictions)})
        del model
        variants[variant] = {"selected": min(trials, key=lambda trial: trial["dev_wer"]), "trials": trials}
    write_json(args.work / "selection.json", {
        "base_revision": REVISION, "dev_sha256": training["dev_sha256"], "dev_count": len(clips),
        "adapter": adapter.name, "adapter_sha256": file_hash(adapter / "adapter_model.safetensors"),
        "batch_size": args.batch_size, "variants": variants,
    })


def predict(args):
    if args.split == "test":
        selected = json.loads((args.work / "selection.json").read_text(encoding="utf-8"))
        beams = selected["variants"][args.variant]["selected"]["num_beams"]
        adapter = args.work / selected["adapter"] if args.variant == "finetuned" else None
        if adapter and file_hash(adapter / "adapter_model.safetensors") != selected["adapter_sha256"]:
            raise ValueError("The selected checkpoint changed after validation.")
        batch_size = selected["batch_size"]
        stem = f"whisper_small_{args.variant}"
        csv_path = ROOT / "data" / "predictions" / f"{stem}.csv"
        manifest_path = ROOT / "results" / "whisper_small" / f"{stem}_run_manifest.json"
    else:
        beams, adapter, batch_size = 1, None, args.batch_size
        csv_path = args.work / "whisper_small_smoke.csv"
        manifest_path = args.work / "whisper_small_smoke_run_manifest.json"
    if csv_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Run output already exists: {csv_path} or {manifest_path}")
    clips = read_split(args.split, limit=args.count if args.split == "validation" else None)
    processor, model, device, dtype = model_setup(args, adapter=adapter)
    # Warm the device before timing using synthetic input rather than a test clip.
    with torch.inference_mode():
        silence = torch.zeros((1, 80, 3000), device=device, dtype=dtype)
        model.generate(silence, attention_mask=torch.ones((1, 3000), device=device),
                       language=LANGUAGE, task=TASK, max_new_tokens=1)
    rows = decode(model, processor, clips, device, dtype, batch_size, beams)
    run = manifest(args.split, device, dtype, batch_size, beams, clips, adapter)
    if args.split == "test":
        run["validation_selection"] = selected["variants"][args.variant]
        run["validation_utterances"] = selected["dev_count"]
        run["validation_split_sha256"] = selected["dev_sha256"]
        if adapter is not None:
            training = json.loads((args.work / "training.json").read_text(encoding="utf-8"))
            run["fine_tuning"] = {key: training[key] for key in (
                "method", "seed", "train_count", "excluded_over_30_seconds",
                "train_sha256", "batch_size", "learning_rate", "optimizer",
                "history", "best_checkpoint",
            )}
    write_predictions(csv_path, rows)
    write_json(manifest_path, run)
    print(f"Predictions: {csv_path}\nRun Manifest: {manifest_path}", flush=True)


def score(_args):
    from src.evaluation.evaluate import evaluate_model, load_references

    references = load_references()
    for variant in ("baseline", "finetuned"):
        stem = f"whisper_small_{variant}"
        result = evaluate_model(ROOT / "data" / "predictions" / f"{stem}.csv", references)
        write_json(ROOT / "results" / "whisper_small" / f"{stem}_metrics.json", result)
        print(f"{variant}: normalized WER {result['normalized']['WER']:.2%}", flush=True)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, help="Local snapshot of the pinned base model")
    parser.add_argument("--work", type=Path, required=True, help="Directory for checkpoints and validation records")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--batch-size", type=positive_int, default=8)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--epochs", type=positive_int, default=2)
    training.add_argument("--train-size", type=positive_int, help="Optional seeded subset for a smaller experiment")
    training.add_argument("--dev-size", type=positive_int, default=384)
    training.add_argument("--learning-rate", type=float, default=2e-4)
    commands.add_parser("select")
    inference = commands.add_parser("predict")
    inference.add_argument("--split", choices=("validation", "test"), required=True)
    inference.add_argument("--variant", choices=("baseline", "finetuned"), default="baseline")
    inference.add_argument("--count", type=positive_int, default=8, help="Validation smoke-test count")
    commands.add_parser("score")
    args = parser.parse_args()
    if args.command == "predict" and args.split == "validation" and args.variant != "baseline":
        parser.error("The validation smoke test uses the baseline.")
    {"train": train, "select": select, "predict": predict, "score": score}[args.command](args)


if __name__ == "__main__":
    main()
