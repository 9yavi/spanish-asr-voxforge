# Whisper-small contribution

This contribution evaluates the multilingual [Whisper-small checkpoint](https://huggingface.co/openai/whisper-small/tree/973afd24965f72e36ca33b3055d56a652f456b4d) in Spanish transcription mode. It also fine-tunes low-rank adapters on the query and value projections of Whisper's decoder attention layers.

- Code: `src/models/whisper_small.py`
- Dependencies: `src/models/requirements-whisper.txt`
- Final raw transcripts: `data/predictions/whisper_small_baseline.csv` and `data/predictions/whisper_small_finetuned.csv`
- Run manifests and evaluation: `results/whisper_small/`

## Reproduce the experiment

From the repository root, with the VoxForge WAV files materialized by Git LFS and a CUDA-enabled PyTorch environment, install the model-specific dependencies:

```console
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r src/models/requirements-whisper.txt
```

```console
python -m src.models.whisper_small --work artifacts/whisper_small predict --split validation --count 8
python -m src.models.whisper_small --work artifacts/whisper_small --batch-size 24 train
python -m src.models.whisper_small --work artifacts/whisper_small select
python -m src.models.whisper_small --work artifacts/whisper_small predict --split test --variant baseline
python -m src.models.whisper_small --work artifacts/whisper_small predict --split test --variant finetuned
python -m src.models.whisper_small --work artifacts/whisper_small score
python -m src.evaluation.bootstrap --model-a whisper_small_baseline --model-b whisper_small_finetuned
```

The script downloads the pinned base model from Hugging Face by default. Supply `--base-dir` before the subcommand to use an existing local copy of that revision.

Training uses only `train.jsonl` and excludes recordings over Whisper's 30-second input window. Checkpoint selection uses normalized corpus WER with greedy decoding on a fixed, seeded sample of 384 validation clips. Beam widths 1 and 3 are evaluated on the same validation sample for each variant. Final inference uses the chosen settings and audio only. Timing reports synchronized model generation time divided across each batch.

Checkpoints and validation decisions are stored in the specified work directory. The `score` command calls the team's evaluator and writes Whisper-specific metrics.

## Completed run

The run used an NVIDIA RTX 4060 Laptop GPU with BF16 compute. Training used 15,179 recordings for two epochs with batch size 24 and AdamW at a learning rate of 0.0002. One recording exceeded 30 seconds and was excluded. The adapter has 589,824 trainable parameters. Greedy validation WER was 3.24% at epoch 1 and 2.69% at epoch 2, so epoch 2 was selected. Beam width 3 gave the lowest WER for both variants on the 384 validation clips, with 7.82% for the baseline and 2.35% for the fine-tuned model.

Each variant was then run once on all 3,256 test recordings with beam width 3 and batch size 8. Both prediction CSVs contain exactly one raw transcript per test audio ID.

### Pretrained baseline

| Normalized WER | Normalized CER | RTF |
| ---: | ---: | ---: |
| 8.20% | 3.01% | 0.0172 |

### Team fine-tuning contribution

| Normalized WER | Normalized CER | RTF |
| ---: | ---: | ---: |
| 3.26% | 1.72% | 0.0185 |

Fine-tuning lowered normalized test WER by 4.94 percentage points. Paired bootstrap over 10,000 replicates gave a 95% confidence interval of 3.93 to 5.71 percentage points for the reduction.

The team's test split holds out speakers. Most test transcripts also appear in training, so the test scores measure recognition of new speakers reading largely familiar text.

