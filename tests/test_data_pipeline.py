import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class TestDataPipeline(unittest.TestCase):

    def test_cleaned_metadata_matches_raw(self):
        raw = load_jsonl(ROOT / "data/raw/metadata.jsonl")
        cleaned = load_jsonl(ROOT / "data/cleaned/metadata.jsonl")
        self.assertEqual(raw, cleaned)

    def test_audio_files_exist(self):
        records = load_jsonl(ROOT / "data/cleaned/metadata.jsonl")
        missing = [
            x["audio_id"] for x in records
            if not (ROOT / "data/raw/audio" / f"{x['audio_id']}.wav").is_file()
        ]
        self.assertEqual(missing, [])

    def test_transcripts_are_not_empty(self):
        records = load_jsonl(ROOT / "data/cleaned/metadata.jsonl")
        self.assertTrue(all(
            isinstance(x.get("text"), str) and x["text"].strip()
            for x in records
        ))

    def test_speaker_independent_splits(self):
        train = load_jsonl(ROOT / "data/splits/train.jsonl")
        val = load_jsonl(ROOT / "data/splits/validation.jsonl")
        test = load_jsonl(ROOT / "data/splits/test.jsonl")

        speakers = [
            {x["speaker_id"] for x in split}
            for split in (train, val, test)
        ]

        self.assertTrue(speakers[0].isdisjoint(speakers[1]))
        self.assertTrue(speakers[0].isdisjoint(speakers[2]))
        self.assertTrue(speakers[1].isdisjoint(speakers[2]))


if __name__ == "__main__":
    unittest.main()
