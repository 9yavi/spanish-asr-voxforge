
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import jiwer

from scripts.normalize_text import normalize_text


TEST_FILE = Path("data/splits/test.jsonl")
PREDICTIONS_DIR = Path("data/predictions")
KEY_TERMS_FILE = Path("data/key_terms.txt")
RESULTS_DIR = Path("results")

# Explicit transformations:
# Preserve case and punctuation for raw evaluation.
WORD_TRANSFORM = jiwer.Compose([
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords()
])

CHAR_TRANSFORM = jiwer.Compose([
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfChars()
])


def load_references():
    references = {}

    with TEST_FILE.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            audio_id = record["audio_id"]

            if audio_id in references:
                raise ValueError(f"Duplicate test audio_id: {audio_id}")

            references[audio_id] = {
                "text": record["text"],
                "duration": float(record["duration"])
            }

    return references


def load_predictions(csv_path):
    predictions = {}

    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        required_columns = {
            "audio_id",
            "prediction",
            "inference_time_seconds"
        }

        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Missing required columns in {csv_path}"
            )

        for row in reader:
            audio_id = row["audio_id"]

            if not audio_id:
                raise ValueError("Empty audio_id found.")

            if audio_id in predictions:
                raise ValueError(f"Duplicate prediction: {audio_id}")

            prediction = row["prediction"]

            if prediction is None:
                raise ValueError(f"Missing prediction: {audio_id}")

            inference_time = float(row["inference_time_seconds"])

            if not math.isfinite(inference_time) or inference_time < 0:
                raise ValueError(
                    f"Invalid inference time for {audio_id}"
                )

            predictions[audio_id] = {
                "text": prediction,
                "inference_time": inference_time
            }

    return predictions


def calculate_metrics(references, hypotheses):
    if len(references) != len(hypotheses):
        raise ValueError("Reference and hypothesis counts do not match.")

    return {
        "WER": jiwer.wer(
            references,
            hypotheses,
            reference_transform=WORD_TRANSFORM,
            hypothesis_transform=WORD_TRANSFORM
        ),
        "CER": jiwer.cer(
            references,
            hypotheses,
            reference_transform=CHAR_TRANSFORM,
            hypothesis_transform=CHAR_TRANSFORM
        ),
        "MER": jiwer.mer(
            references,
            hypotheses,
            reference_transform=WORD_TRANSFORM,
            hypothesis_transform=WORD_TRANSFORM
        ),
        "WIL": jiwer.wil(
            references,
            hypotheses,
            reference_transform=WORD_TRANSFORM,
            hypothesis_transform=WORD_TRANSFORM
        )
    }


def load_key_terms():
    if not KEY_TERMS_FILE.exists():
        raise FileNotFoundError(KEY_TERMS_FILE)

    with KEY_TERMS_FILE.open(encoding="utf-8") as f:
        terms = [
            normalize_text(line.strip())
            for line in f
            if line.strip()
        ]

    if not terms:
        raise ValueError("Key terms file is empty.")

    if len(terms) != len(set(terms)):
        raise ValueError("Duplicate key terms found.")

    return terms


def calculate_key_term_recall(references, hypotheses):
    key_terms = load_key_terms()

    total_reference_terms = 0
    correctly_recognized_terms = 0
    term_results = {}

    for term in key_terms:
        reference_count = 0
        recognized_count = 0

        for reference, hypothesis in zip(references, hypotheses):
            ref_words = Counter(normalize_text(reference).split())
            hyp_words = Counter(normalize_text(hypothesis).split())

            reference_count += ref_words[term]

            recognized_count += min(
                ref_words[term],
                hyp_words[term]
            )

        total_reference_terms += reference_count
        correctly_recognized_terms += recognized_count

        term_results[term] = {
            "reference_count": reference_count,
            "recognized_count": recognized_count,
            "recall": (
                recognized_count / reference_count
                if reference_count > 0 else None
            )
        }

    overall_recall = (
        correctly_recognized_terms / total_reference_terms
        if total_reference_terms > 0 else None
    )

    return {
        "overall_recall": overall_recall,
        "recognized": correctly_recognized_terms,
        "total": total_reference_terms,
        "per_term": term_results
    }


def evaluate_model(csv_path, references_data):
    model_name = csv_path.stem

    predictions_data = load_predictions(csv_path)

    expected_ids = set(references_data)
    prediction_ids = set(predictions_data)

    missing = expected_ids - prediction_ids
    extra = prediction_ids - expected_ids

    if missing or extra:
        raise ValueError(
            f"{model_name}: Prediction mismatch. "
            f"Missing: {len(missing)}, Extra: {len(extra)}"
        )

    audio_ids = list(references_data.keys())

    raw_references = [
        references_data[audio_id]["text"]
        for audio_id in audio_ids
    ]

    raw_predictions = [
        predictions_data[audio_id]["text"]
        for audio_id in audio_ids
    ]

    normalized_references = [
        normalize_text(text)
        for text in raw_references
    ]

    normalized_predictions = [
        normalize_text(text)
        for text in raw_predictions
    ]

    raw_metrics = calculate_metrics(
        raw_references,
        raw_predictions
    )

    normalized_metrics = calculate_metrics(
        normalized_references,
        normalized_predictions
    )

    key_term_results = calculate_key_term_recall(
        raw_references,
        raw_predictions
    )

    total_audio_duration = sum(
        references_data[audio_id]["duration"]
        for audio_id in audio_ids
    )

    total_inference_time = sum(
        predictions_data[audio_id]["inference_time"]
        for audio_id in audio_ids
    )

    rtf = (
        total_inference_time / total_audio_duration
        if total_audio_duration > 0 else None
    )

    rtfx = (
        1 / rtf
        if rtf is not None and rtf > 0 else None
    )

    return {
        "model": model_name,
        "utterances": len(audio_ids),
        "raw": raw_metrics,
        "normalized": normalized_metrics,
        "key_term_recall": key_term_results,
        "total_audio_hours": total_audio_duration / 3600,
        "total_inference_seconds": total_inference_time,
        "RTF": rtf,
        "RTFx": rtfx
    }


def save_results(results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Main comparison table
    output_file = RESULTS_DIR / "evaluation_results.csv"

    rows = []

    for result in results:
        row = {
            "model": result["model"],
            "utterances": result["utterances"],
            "total_audio_hours": result["total_audio_hours"],
            "total_inference_seconds": result["total_inference_seconds"],
            "RTF": result["RTF"],
            "RTFx": result["RTFx"],
            "key_term_recall": result["key_term_recall"]["overall_recall"],
            "recognized_key_terms": result["key_term_recall"]["recognized"],
            "total_key_terms": result["key_term_recall"]["total"]
        }

        for metric, value in result["raw"].items():
            row[f"raw_{metric}"] = value

        for metric, value in result["normalized"].items():
            row[f"normalized_{metric}"] = value

        rows.append(row)

    if rows:
        with output_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    # Detailed key-term results
    key_term_file = RESULTS_DIR / "key_term_recall.csv"

    term_rows = []

    for result in results:
        for term, values in result["key_term_recall"]["per_term"].items():
            term_rows.append({
                "model": result["model"],
                "term": term,
                "reference_count": values["reference_count"],
                "recognized_count": values["recognized_count"],
                "recall": values["recall"]
            })

    if term_rows:
        with key_term_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=term_rows[0].keys())
            writer.writeheader()
            writer.writerows(term_rows)

    print(f"\nMain results saved to: {output_file}")
    print(f"Key-term results saved to: {key_term_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Spanish ASR model predictions."
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Prediction CSV filename without extension."
    )

    args = parser.parse_args()

    if args.model:
        csv_files = [
            PREDICTIONS_DIR / f"{args.model}.csv"
        ]
    else:
        csv_files = sorted(
            path for path in PREDICTIONS_DIR.glob("*.csv")
            if path.name != "template.csv"
        )

    if not csv_files:
        raise FileNotFoundError(
            "No model prediction CSV files found."
        )

    references_data = load_references()

    results = []

    for csv_path in csv_files:
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)

        print(f"\nEvaluating: {csv_path.stem}")

        result = evaluate_model(csv_path, references_data)
        results.append(result)

        print("Raw metrics:", result["raw"])
        print("Normalized metrics:", result["normalized"])

        print(
            "Key-Term Recall:",
            result["key_term_recall"]["overall_recall"]
        )

        print(f"RTF: {result['RTF']}")
        print(f"RTFx: {result['RTFx']}")

    save_results(results)


if __name__ == "__main__":
    main()
