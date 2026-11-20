
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import jiwer

from src.evaluation.normalize_text import normalize_text


ROOT = Path(__file__).resolve().parents[2]
TEST_FILE = ROOT / "data/splits/test.jsonl"
PREDICTIONS_DIR = ROOT / "data/predictions"


def load_references():
    references = {}

    with TEST_FILE.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            references[record["audio_id"]] = record["text"]

    return references


def load_predictions(model_name):
    path = PREDICTIONS_DIR / f"{model_name}.csv"
    predictions = {}

    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        required = {"audio_id", "prediction"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Invalid prediction CSV: {path}")

        for row in reader:
            audio_id = row["audio_id"]

            if audio_id in predictions:
                raise ValueError(f"Duplicate audio_id: {audio_id}")

            predictions[audio_id] = row["prediction"]

    return predictions


def get_word_errors(reference, hypothesis):
    result = jiwer.process_words(
        [reference],
        [hypothesis]
    )

    errors = (
        result.substitutions
        + result.deletions
        + result.insertions
    )

    reference_words = (
        result.hits
        + result.substitutions
        + result.deletions
    )

    return errors, reference_words


def prepare_model_data(references, predictions, audio_ids):
    errors = []
    word_counts = []

    for audio_id in audio_ids:
        reference = normalize_text(references[audio_id])
        hypothesis = normalize_text(predictions[audio_id])

        error_count, reference_word_count = get_word_errors(
            reference,
            hypothesis
        )

        errors.append(error_count)
        word_counts.append(reference_word_count)

    return np.array(errors), np.array(word_counts)


def calculate_wer(errors, word_counts, indices=None):
    if indices is not None:
        errors = errors[indices]
        word_counts = word_counts[indices]

    total_words = word_counts.sum()

    if total_words == 0:
        return 0.0

    return errors.sum() / total_words


def paired_bootstrap(
    errors_a,
    words_a,
    errors_b,
    words_b,
    iterations=10000,
    seed=42
):
    rng = np.random.default_rng(seed)
    sample_count = len(errors_a)

    differences = []

    for _ in range(iterations):
        indices = rng.integers(
            0,
            sample_count,
            size=sample_count
        )

        wer_a = calculate_wer(errors_a, words_a, indices)
        wer_b = calculate_wer(errors_b, words_b, indices)

        differences.append(wer_a - wer_b)

    differences = np.array(differences)

    lower = np.percentile(differences, 2.5)
    upper = np.percentile(differences, 97.5)

    return lower, upper


def main():
    parser = argparse.ArgumentParser(
        description="Paired Bootstrap comparison of two ASR models."
    )

    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.iterations < 1:
        raise ValueError("Iterations must be positive.")

    references = load_references()

    predictions_a = load_predictions(args.model_a)
    predictions_b = load_predictions(args.model_b)

    audio_ids = list(references.keys())
    expected_ids = set(audio_ids)

    if set(predictions_a) != expected_ids:
        raise ValueError(f"{args.model_a}: Test IDs do not match.")

    if set(predictions_b) != expected_ids:
        raise ValueError(f"{args.model_b}: Test IDs do not match.")

    errors_a, words_a = prepare_model_data(
        references, predictions_a, audio_ids
    )

    errors_b, words_b = prepare_model_data(
        references, predictions_b, audio_ids
    )

    wer_a = calculate_wer(errors_a, words_a)
    wer_b = calculate_wer(errors_b, words_b)

    delta = wer_a - wer_b

    lower, upper = paired_bootstrap(
        errors_a,
        words_a,
        errors_b,
        words_b,
        iterations=args.iterations,
        seed=args.seed
    )

    print("\nPaired Bootstrap Results")
    print("------------------------")
    print(f"Model A: {args.model_a}")
    print(f"Model B: {args.model_b}")
    print(f"Utterances: {len(audio_ids)}")
    print(f"Iterations: {args.iterations}")
    print(f"Normalized WER A: {wer_a:.4f}")
    print(f"Normalized WER B: {wer_b:.4f}")
    print(f"WER difference (A - B): {delta:.4f}")
    print(f"95% Confidence Interval: [{lower:.4f}, {upper:.4f}]")

    if lower > 0 or upper < 0:
        print("Result: The WER difference is statistically significant at 5%.")
    else:
        print("Result: No statistically significant difference detected at 5%.")


if __name__ == "__main__":
    main()
