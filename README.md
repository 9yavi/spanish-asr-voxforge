# Spanish Automatic Speech Recognition (ASR)

A Spanish Automatic Speech Recognition project using the VoxForge Spanish dataset. The project compares traditional acoustic models, neural ASR architectures, and pretrained speech recognition systems under a consistent speaker-independent evaluation setup.

All evaluated systems use the same test recordings, reference transcripts, text normalization, and evaluation metrics to ensure a fair comparison.

## 1. Project Overview

The objective is to evaluate different approaches to Spanish speech recognition, including:

- Traditional acoustic models.
- Neural CTC-based architectures.
- Attention encoder-decoder architectures.
- Transducer-based architectures.
- Pretrained multilingual and Spanish-capable ASR models.
- Team fine-tuning experiments.

The project separates pretrained baselines from models fine-tuned by the team.

## 2. Dataset

**Dataset:** [VoxForge Spanish](https://huggingface.co/datasets/ciempiess/voxforge_spanish)

The dataset contains Spanish speech recordings with reference transcripts and speaker metadata.

### Dataset Statistics

| Property | Value |
|---|---:|
| Total recordings | 21,692 |
| Unique speakers | 2,180 |
| Total audio duration | 49.71 hours |
| Sample rate | 16,000 Hz |
| Audio channels | Mono |
| Missing audio files | 0 |
| Corrupt audio files | 0 |

All extracted recordings were verified as readable, non-empty, 16 kHz, mono WAV files. The source audio already met these specifications, so no resampling was necessary.

### Dataset Distribution

The dataset includes speakers from different Spanish-speaking regions, including Spain, Argentina, Mexico, Chile, and other Latin American regions.

## 3. Repository Structure

```text
spanish-asr-voxforge/
│
├── data/
│   ├── raw/
│   │   ├── audio/
│   │   └── metadata.jsonl
│   ├── cleaned/
│   │   ├── metadata.jsonl
│   │   └── cleaning_report.json
│   ├── splits/
│   │   ├── train.jsonl
│   │   ├── validation.jsonl
│   │   └── test.jsonl
│   ├── predictions/
│   │   └── template.csv
│   └── key_terms.txt
│
├── src/
│   ├── data/
│   ├── evaluation/
│   └── models/
│
├── tests/
├── docs/
├── results/
├── contributions/
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

- `data/raw/`: Original dataset files and metadata.
- `data/cleaned/`: Validated metadata produced by the cleaning pipeline.
- `data/splits/`: Train, validation, and test manifests.
- `data/predictions/`: Model prediction files.
- `src/data/`: Data inspection, validation, cleaning, extraction, and splitting scripts.
- `src/evaluation/`: Text normalization, evaluation, and statistical analysis.
- `src/models/`: Model implementations and experiments.
- `tests/`: Automated tests for the data pipeline.
- `docs/`: Dataset and preprocessing documentation.
- `results/`: Evaluation outputs.
- `contributions/`: Individual team member contribution records.

## 4. Data Preparation and Cleaning

The data preparation pipeline includes:

1. Dataset extraction and local storage.
2. Audio file inspection and validation.
3. Duplicate audio detection using SHA-256 hashes.
4. Transcript inspection and normalization checks.
5. Speaker-independent dataset splitting.
6. Silence and non-speech analysis using WebRTC VAD.

### Cleaning Results

The validation process inspected all 21,692 recordings.

| Check | Result |
|---|---:|
| Missing audio | 0 |
| Corrupt audio | 0 |
| Invalid sample rate or channels | 0 |
| Empty transcripts | 0 |
| Exact duplicate audio groups | 0 |
| Excluded recordings | 0 |
| Transcript normalization changes | 0 |

No audio corrections or record exclusions were necessary based on these checks. The original audio files were retained unchanged.

The cleaned metadata is stored in `data/cleaned/metadata.jsonl`. Its audio paths reference the original files under `data/raw/audio/`.

### Repeated Transcripts

The analysis identified 44 groups of repeated normalized transcripts, involving 21,687 recordings.

These recordings were retained because identical text does not imply identical audio. They may represent different speakers reading the same prompts.

The transcript-to-audio semantic alignment has not been manually reviewed for every recording.

More details: `docs/cleaning_report.md`.

## 5. Dataset Splitting

The dataset was divided by speaker to prevent speakers from appearing in multiple splits.

Random seed: `42`

| Split | Recordings | Speakers | Duration |
|---|---:|---:|---:|
| Train | 15,180 | 1,526 | 34.76 hours |
| Validation | 3,256 | 327 | 7.43 hours |
| Test | 3,256 | 327 | 7.52 hours |
| **Total** | **21,692** | **2,180** | **49.71 hours** |

There is no speaker overlap between the three splits.

### Text Overlap Limitation

A comparison of normalized transcripts found that 3,255 of the 3,256 test transcripts also occur in the training set, across 43 unique matching texts.

This is transcript overlap, not speaker leakage. The evaluation therefore measures recognition on unseen speakers, but does not establish generalization to entirely unseen sentences.

No session identifier was available in the inspected metadata, so session-level grouping could not be performed.

## 6. Evaluation Methodology

All models must be evaluated using the same test recordings and reference transcripts.

The evaluation pipeline reports:

- Word Error Rate (WER)
- Character Error Rate (CER)
- Match Error Rate (MER)
- Word Information Lost (WIL)
- Key-term recall
- Inference time
- Real-Time Factor (RTF)
- Real-Time Factor speed (RTFx)

Text normalization applies:

- Unicode NFC normalization.
- Lowercasing.
- Punctuation removal.
- Whitespace normalization.

Spanish accents and `ñ` are preserved, and digits are retained.

WER and CER are the primary comparison metrics. MER and WIL provide additional error perspectives.

Evaluation methodology reference: [Hugging Face Audio Course – Evaluation](https://huggingface.co/learn/audio-course/en/chapter5/evaluation).

### Prediction File Format

Model predictions must be provided as CSV files with these columns:

| Column | Description |
|---|---|
| `audio_id` | Test recording identifier |
| `prediction` | Model-generated transcript |
| `inference_time_seconds` | Inference time for the recording |

A template is available at `data/predictions/template.csv`.

Each prediction file must contain exactly one prediction for every test recording.

## 7. Statistical Analysis

The project includes paired utterance-level bootstrap analysis to estimate 95% confidence intervals.

The bootstrap implementation uses 10,000 iterations and random seed `42`.

Statistical comparisons should use predictions from the same test utterances.
### Running Paired Bootstrap Comparison

To compare two ASR models using paired utterance-level bootstrap:

```bash
python src/evaluation/bootstrap.py \
  --model-a MODEL_A \
  --model-b MODEL_B
```

Replace `MODEL_A` and `MODEL_B` with prediction filenames without the `.csv` extension.

Example:

```bash
python src/evaluation/bootstrap.py \
  --model-a whisper_small \
  --model-b wav2vec2
```

Optional arguments:

| Argument | Default | Description |
|---|---:|---|
| `--iterations` | 10000 | Number of bootstrap iterations |
| `--seed` | 42 | Random seed for reproducibility |

Example with custom settings:

```bash
python src/evaluation/bootstrap.py \
  --model-a whisper_small \
  --model-b wav2vec2 \
  --iterations 10000 \
  --seed 42
```

The script reports both models' normalized WER, their difference (A − B), and a 95% bootstrap confidence interval.

A confidence interval that does not include zero indicates a statistically significant difference at the 5% level, under the assumptions of the paired bootstrap procedure.

## 8. Models and Baselines

All results below use the same 3,256 test utterances, reference transcripts, and text normalization. WER measures word edits relative to reference word count. CER measures character edits, including errors on accented letters. Lower is better.

| Baseline checkpoint | Revision | WER ↓ | CER ↓ |
|---|---|---:|---:|
| `openai/whisper-small` | `973afd24965f72e36ca33b3055d56a652f456b4d` | 8.20% | 3.01% |
| `Qwen/Qwen3-ASR-0.6B` | `5eb144179a02acc5e5ba31e748d22b0cf3e303b0` | 7.51% | 2.82% |
| `nvidia/canary-1b-v2` | `d455706339a6b32e1aa40f82c713a482a0c938e2` | 4.61% | 1.89% |
| `nvidia/parakeet-tdt-0.6b-v3` | `541d1f99c6b0c3cd0b11a95167540bb8edefd82b` | 4.27% | 1.74% |
| `jonatasgrosman/wav2vec2-large-xlsr-53-spanish` | `96d7e9b4e4a78af515a3c6d3cee7c0826045d276` | 8.52% | 2.47% |

`openai/whisper-small` is the multilingual Whisper checkpoint used for Spanish transcription. See the [generated evaluation CSV](results/evaluation_results.csv) for the full metrics and the [multilingual baseline run notes](contributions/member2_baselines.md) for environment and inference commands.

The [wav2vec 2.0 run notes](results/wav2vec2_spanish/run_notes.md) document Spanish CTC greedy decoding.

**Not evaluated in this benchmark:** GMM-HMM with MFCC or log-mel features, standalone CTC encoder, attention encoder-decoder or hybrid CTC/attention, standalone RNN-T, and Spanish HuBERT + CTC.

Pretrained models evaluated without additional team training are reported as baselines. Team fine-tuning experiments are reported separately as project contributions.

## 9. Training and Experiments

| Team contribution | WER ↓ | CER ↓ |
|---|---:|---:|
| `openai/whisper-small` + decoder-attention LoRA (`epoch-2`) | 3.26% | 1.72% |

The validation split is used for checkpoint selection and decoding configuration. The test split is reserved for final evaluation.

We trained LoRA for two epochs on 15,179 training clips, excluding one clip longer than 30 seconds. We used seed 42 and learning rate `2e-4`. We selected the adapter and decoding settings using a fixed sample of 384 validation clips. The WER reduction versus the unmodified Whisper checkpoint was 4.94 percentage points on this test set.

Paired utterance bootstrap (10,000 replicates, seed 42) gave a 95% confidence interval of 3.93 to 5.71 percentage points for that WER reduction. See the [Whisper run notes](contributions/whisper_small.md) for the training and inference commands.

## 10. Installation

### Requirements

- Python 3.13
- Git
- Git LFS
- uv

### Clone the Repository

```bash
git lfs install
git clone https://github.com/9yavi/spanish-asr-voxforge.git
cd spanish-asr-voxforge
git lfs pull
```

### Create a Virtual Environment

```bash
uv venv --python 3.13 .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
uv pip install -r requirements.txt
```


## 11. Running the Data Pipeline

Run commands from the repository root with the virtual environment activated.

### Inspect the Dataset

```bash
python src/data/inspect_dataset.py
```

### Analyze the Dataset

```bash
python src/data/analyze_dataset.py
```

### Clean and Validate Metadata

```bash
python src/data/clean_dataset.py
```

### Create Speaker-Independent Splits

```bash
python src/data/split_dataset.py
```

### Analyze Silence and Non-Speech

```bash
python src/data/analyze_silence_vad.py
```

### Extract Audio

```bash
python src/data/extract_audio.py
```

Run extraction only when preparing the dataset from its original source. It writes the extracted audio and metadata to the raw data directory.

## 12. Running Tests

The data pipeline tests use Python's standard `unittest` framework.

```bash
python -m unittest discover -s tests -v
```

## 13. Running Evaluation

Model predictions must be saved as CSV files under `data/predictions/`, following the required prediction format.

### Evaluate All Models

```bash
python -m src.evaluation.evaluate
```

This evaluates all prediction CSV files in `data/predictions/`, excluding `template.csv`.

### Evaluate a Specific Model

```bash
python -m src.evaluation.evaluate --model MODEL_NAME
```

Replace `MODEL_NAME` with the prediction filename without the `.csv` extension.

For example:

```bash
python -m src.evaluation.evaluate --model whisper_small_baseline
```

The evaluation pipeline checks prediction IDs against the test set and generates:

- `results/evaluation_results.csv`
- `results/key_term_recall.csv`

The outputs include WER, CER, MER, WIL, key-term recall, inference time, RTF, and RTFx.

### Statistical Analysis

Paired utterance-level bootstrap analysis is implemented in:

```text
src/evaluation/bootstrap.py
```

It uses 10,000 iterations and random seed `42` to estimate 95% confidence intervals.

## 14. Reproducibility

The project aims to keep data preparation and evaluation reproducible through:

- Version-controlled scripts.
- Fixed random seeds.
- Documented dependencies.
- Speaker-independent dataset splits.
- A common evaluation procedure.
- Automated data pipeline tests.

Model-specific installation and inference commands will be documented alongside their implementations.

## 15. Current Limitations

- Repeated transcript text occurs across different speakers in the dataset.
- The test set is speaker-independent but not text-independent.
- Session identifiers were not available in the inspected metadata.
- Transcript-to-audio semantic alignment has not been manually verified for every recording.
- Timing protocols differ across models. We exclude speed from the cross-model comparison.

## 16. Team Contributions

Each team member is responsible for documenting their work in a separate file under `contributions/`.

Contribution records must identify:

- Implemented work.
- Relevant files.
- Experiments performed.
- Results obtained.
- Commits made by that team member.

## 17. References

- [VoxForge Spanish Dataset](https://huggingface.co/datasets/ciempiess/voxforge_spanish)
- [Hugging Face Audio Course – Evaluation](https://huggingface.co/learn/audio-course/en/chapter5/evaluation)
