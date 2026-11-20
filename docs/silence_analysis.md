# Silence and Non-Speech Analysis

## Methodology

- Tool: WebRTC Voice Activity Detection (VAD)
- Sample Rate: 16 kHz
- Frame Duration: 20 ms
- VAD Aggressiveness Mode: 2
- A new VAD instance was created for each recording.
- Non-speech frames were identified and aggregated per recording.

## Overall Results

- Total recordings: 21,692
- Total analyzed duration: 49.67 hours
- Estimated non-speech duration: 13.31 hours
- Overall non-speech percentage: 26.80%
- Recordings containing non-speech: 20,962

## Distribution

| Non-Speech Range | Recordings | Percentage |
|---|---:|---:|
| 0-10% | 1,548 | 7.14% |
| 10-20% | 3,420 | 15.77% |
| 20-30% | 8,231 | 37.94% |
| 30-40% | 6,203 | 28.60% |
| 40-50% | 1,867 | 8.61% |
| 50-100% | 423 | 1.95% |

## Statistical Summary

- Mean per-recording non-speech: 27.02%
- Median: 27.36%
- Standard deviation: 11.19%
- Minimum: 0%
- Maximum: 83.57%

## Important Limitation

WebRTC VAD classifies audio frames as speech or non-speech.
Non-speech does not necessarily mean silence. Low-energy speech
may also be classified as non-speech.

Therefore, these results are estimates of VAD-detected non-speech,
not verified measurements of actual silence.

## Output

- `data/splits/silence_analysis_vad.csv`