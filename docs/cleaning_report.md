# Dataset Cleaning Report

## Dataset
- Source: `ciempiess/voxforge_spanish`
- Total records: 21,692
- Unique speakers: 2,180
- Total duration: 49.71 hours

## Audio Validation
All 21,692 audio files were checked.
- Missing audio files: 0
- Corrupt audio files: 0
- Invalid sample rate or channel count: 0
- Audio format: WAV, 16 kHz, mono
- Audio files modified: 0

The source audio was already 16 kHz mono. No resampling was required.

## Transcript Validation
- Empty transcripts: 0
- Transcript normalization changes: 0
- Exact duplicate audio groups: 0
- Records excluded: 0

Transcripts were checked and normalized using Unicode NFC and whitespace normalization.

## Repeated Text Analysis
The dataset contains 44 groups of repeated normalized transcripts, involving 21,687 records.

These records contain different audio recordings, not duplicate audio. The repeated text was retained.

The speaker-independent split has no speaker overlap. However, 3,255 of the 3,256 Test transcripts also appear in Train, across 43 unique matching texts.

Therefore, the evaluation measures speaker-independent recognition, not generalization to unseen sentences.

## Cleaning Output
The cleaning script creates:
- `data/cleaned/metadata.jsonl`
- `data/cleaned/cleaning_report.json`

The cleaned metadata references the original audio files in `data/raw/audio/`. Audio files were not modified or duplicated because validation found no issues requiring correction.

## Data Splits
| Split | Records | Speakers | Duration |
|---|---:|---:|---:|
| Train | 15,180 | 1,526 | 34.76 hours |
| Validation | 3,256 | 327 | 7.43 hours |
| Test | 3,256 | 327 | 7.52 hours |

Splits were generated using speaker IDs and a fixed random seed of 42. No speaker appears in more than one split.

## Limitations
- Transcript/audio semantic alignment was not manually reviewed.
- Repeated transcript text across splits limits evaluation of unseen-sentence generalization.
- Audio validation confirms file integrity and format, not recording quality or transcription correctness.
