# Spanish wav2vec2 baseline

Checkpoint: jonatasgrosman/wav2vec2-large-xlsr-53-spanish, revision 96d7e9b4e4a78af515a3c6d3cee7c0826045d276.

CTC greedy decoding used batch size 1 and bfloat16. On 128 validation recordings the run processed 996.725 seconds of audio in 33.15 seconds. Test used all 3,256 speaker-independent recordings and achieved normalized WER 8.5163% and CER 2.4730% using the project evaluator. The project documents that 3,255 test transcripts also appear in training.

Dependencies: Python 3.12.14, PyTorch 2.10.0+cu126, Transformers 4.57.6, soundfile 0.14.0, jiwer 4.0.0.

Run from the repository root, with the snapshot downloaded locally and WAV files materialized:

    python -m src.models.wav2vec2_spanish --model-dir PATH_TO_SNAPSHOT --data-root PATH_TO_PROJECT --split validation --limit 128 --output PATH_TO_VALIDATION_CSV
    python -m src.models.wav2vec2_spanish --model-dir PATH_TO_SNAPSHOT --data-root PATH_TO_PROJECT --split test --output data/predictions/wav2vec2_spanish.csv
    python -m src.evaluation.evaluate --model wav2vec2_spanish

The evaluator writes common results files. Use a separate checkout if preserving existing tables.
