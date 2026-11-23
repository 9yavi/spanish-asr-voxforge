# Member 2 multilingual baselines

These run notes cover the pretrained Qwen3-ASR, Canary, and Parakeet experiments. The [Whisper run notes](whisper_small.md) document the Whisper baseline and the team's LoRA fine-tuning.

| Model | Run name | Language setting |
| --- | --- | --- |
| [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | `qwen3_asr_06b` | Spanish |
| [Canary 1B v2](https://huggingface.co/nvidia/canary-1b-v2) | `canary_1b_v2` | Spanish source and target |
| [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | `parakeet_tdt_06b_v3` | Automatic language detection |

## Reproduce the runs

Use the team's existing 16 kHz mono WAV files and split manifests. Install CUDA-enabled PyTorch and create separate Python environments for the Qwen and NVIDIA backends. The Windows run uses the official Transformers checkpoints for the NVIDIA models. Qwen uses `qwen-asr` with its Transformers backend.

- Qwen environment requires `qwen-asr==0.0.6`, `soundfile`, and `jiwer==4.0.0`.
- NVIDIA environment requires `transformers==5.23.0`, `accelerate==1.12.0`, `soundfile`, and `jiwer==4.0.0`.

Retrieve the snapshots below into `../.task2-work/models/<run-name>` from the repository root. Download `model.safetensors` and the processor and tokenizer files (`*.json`, `*.txt`, and `*.jinja`). The runner checks the published weight digests.

| Run name | Hugging Face revision |
| --- | --- |
| `qwen3_asr_06b` | `5eb144179a02acc5e5ba31e748d22b0cf3e303b0` |
| `canary_1b_v2` | `d455706339a6b32e1aa40f82c713a482a0c938e2` |
| `parakeet_tdt_06b_v3` | `541d1f99c6b0c3cd0b11a95167540bb8edefd82b` |

Activate the environment for the selected model and replace `RUN_NAME` with its run name.

```console
python -m src.models.member2_baselines RUN_NAME validation
python -m src.models.member2_baselines RUN_NAME test
python -m src.models.member2_baselines RUN_NAME score
```

The validation command runs eight clips spanning the duration range in `validation.jsonl`. Test inference uses the checkpoint and settings verified on those clips. It writes raw predictions to `data/predictions/<run-name>.csv`, and writes the Run Manifest and team evaluator metrics to `results/member2/`. The inference clock includes audio feature extraction and text decoding, which differs from the Whisper run's generation-only timing.

## Test results

The tests use the team's 3,256 held-out VoxForge recordings. WER and CER are normalized with the team's Spanish text rules.

| Pretrained model | Normalized WER | Normalized CER |
| --- | ---: | ---: |
| Qwen3-ASR-0.6B | 7.51% | 2.82% |
| Canary 1B v2 | 4.61% | 1.89% |
| Parakeet TDT 0.6B v3 | 4.27% | 1.74% |

All three runs used batch size 8 and BF16 on an NVIDIA RTX 4060 Laptop GPU. Each prediction file contains one row for every test audio ID, in the order of the test manifest.
