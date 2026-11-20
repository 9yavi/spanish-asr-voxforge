
# Dataset Analysis Report

## 1. Dataset Information

- **Dataset Name:** ciempiess/voxforge_spanish
- **Source:** Hugging Face
- **Task:** Spanish Automatic Speech Recognition (ASR)
- **Total Audio Samples:** 21,692
- **Unique Speakers:** 2,180
- **Total Audio Duration:** 49.71 hours
- **Source Files:** 8 Parquet files

## 2. Country Distribution

| Country | Number of Samples |
|---|---:|
| Spain | 16,427 |
| Argentina | 1,735 |
| Latin America | 1,570 |
| Mexico | 758 |
| Chile | 719 |
| Unknown | 483 |
| **Total** | **21,692** |

## 3. Gender Distribution

| Gender | Number of Samples |
|---|---:|
| Male | 17,053 |
| Female | 4,639 |
| **Total** | **21,692** |

## 4. Dataset Observations

- The dataset contains recordings from 2,180 unique speakers.
- Spain accounts for the largest number of recordings.
- Male recordings represent approximately 78.6% of the dataset.
- Female recordings represent approximately 21.4% of the dataset.
- The gender distribution is imbalanced.
- The country distribution is also imbalanced.

## 5. Analysis Method

The dataset was loaded using the Hugging Face `datasets` library.

The analysis was performed locally after downloading the dataset.

The following statistics were calculated:
- Total number of audio samples.
- Number of unique speakers.
- Total audio duration.
- Sample distribution by country.
- Sample distribution by gender.

## 6. Next Steps

1. Extract audio files into `data/raw`.
2. Inspect audio quality and metadata.
3. Convert audio to 16 kHz mono WAV.
4. Check for corrupted or duplicate recordings.
5. Create speaker-independent Train, Development, and Test splits.
6. Implement the evaluation pipeline.

---

**Status:** Initial dataset download and analysis completed.
