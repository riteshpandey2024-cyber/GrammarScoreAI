# Dataset Information

## Overview
This project uses spoken English audio recordings for grammar assessment.

## Dataset Structure
```
Dataset_Final/
├── train/           # Training audio files (.wav)
├── test/            # Test audio files (.wav)
├── train.csv        # Training labels (filename, label)
└── test.csv         # Test metadata (filename)
```

## Data Format
- **Audio Format**: WAV, 16kHz sampling rate
- **Duration**: 45-60 seconds per clip
- **Labels**: Continuous grammar scores [0.0 - 5.0]

## How to Obtain Dataset
> **Note**: Due to size constraints, datasets are not included in this repository.

Contact the project maintainer or refer to the original SHL assessment challenge for access.

## Quick Start
Once you have the dataset:
1. Place files in `Dataset_Final/` directory
2. Ensure structure matches above
3. Run the Jupyter notebook

## Data Statistics
- Training samples: ~XXX audio files
- Test samples: ~XXX audio files
- Total size: ~X GB
