# Dataset Information

## Overview

This project uses Arabic sentiment analysis data for training and evaluation.

## Dataset Details

- **Language:** Arabic
- **Task:** Binary Sentiment Classification (Positive/Negative)
- **Total Samples:** ~4,200
- **Source:** [Arabic Sentiment Analysis Dataset - SS2030](https://www.kaggle.com/datasets/...)

## Dataset Structure

### Expected Format

The dataset should be a CSV file with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `text` | string | Arabic text (tweets, reviews, comments) |
| `Sentiment` | int | Sentiment label (0 = Negative, 1 = Positive) |

### Example Rows
```csv
text,Sentiment
"هذا المنتج رائع جداً وأنصح به بشدة",1
"تجربة سيئة جداً ولن أشتري مرة أخرى",0
"الخدمة ممتازة والتوصيل سريع",1
```

## Download Instructions

Due to file size and licensing restrictions, the dataset is **not included** in this repository.

### Option 1: Download from Kaggle
```bash
# Install Kaggle CLI
pip install kaggle

# Configure API credentials (get from https://www.kaggle.com/settings)
# Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\Users\<username>\.kaggle\ (Windows)

# Download dataset
kaggle datasets download -d DATASET_ID
unzip DATASET_ID.zip -d data/raw/
```

### Option 2: Manual Download

1. Visit: [Dataset Link](https://www.kaggle.com/datasets/...)
2. Click "Download" button
3. Extract CSV file to `data/raw/`
4. Rename to `arabic_sentiment_data.csv` (if needed)

### Option 3: Use Your Own Data

If you have your own Arabic sentiment dataset:

1. Prepare CSV with columns: `text`, `Sentiment`
2. Place in `data/raw/`
3. Update the path in the notebook configuration

## Data Preprocessing

The raw data is preprocessed with the following steps:

1. **Text Cleaning**
   - Remove URLs, mentions, hashtags
   - Remove non-Arabic characters
   - Normalize Arabic characters (alef, yeh, hamza, teh marbuta)
   - Remove diacritics (tashkeel)

2. **Train/Val/Test Split**
   - Training: 72%
   - Validation: 8%
   - Test: 20%
   - Stratified split to maintain class balance

3. **Output**
   - Processed data saved to `data/processed/`
   - Tokenized sequences for model training

## Directory Structure
```
data/
├── README.md                          # This file
│
├── raw/                               # Original downloaded data
│   ├── .gitkeep                      # Keeps folder in Git
│   └── arabic_sentiment_data.csv     # Your dataset (not in Git)
│
└── processed/                         # Preprocessed data (generated)
    ├── .gitkeep
    ├── train.csv                      # Training split
    ├── val.csv                        # Validation split
    └── test.csv                       # Test split
```

## Data Statistics

After preprocessing (approximate):

- **Total Samples:** 4,200
- **Training Samples:** 3,030 (72%)
- **Validation Samples:** 337 (8%)
- **Test Samples:** 842 (20%)

### Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| Negative (0) | ~2,100 | 50% |
| Positive (1) | ~2,100 | 50% |

### Text Length Statistics

- **Average Length:** ~15 words
- **Median Length:** ~12 words
- **Min Length:** 2 words (after filtering)
- **Max Length:** ~100 words

## Data Quality

The preprocessing pipeline ensures:

- ✅ No duplicate texts
- ✅ Minimum text length (≥2 words)
- ✅ Balanced class distribution
- ✅ Clean Arabic text (no mixed scripts)
- ✅ Normalized characters

## Preprocessing Script

To preprocess the data yourself:
```python
# See preprocessing section in notebook
# notebooks/bilstm_vs_arabert_comparison.ipynb

# Key functions:
# - clean_arabic_text()  # Text normalization
# - train_test_split()   # Stratified splitting
```

## Citation

If you use this dataset, please cite the original source:
```bibtex
@dataset{arabic_sentiment_2024,
  title={Arabic Sentiment Analysis Dataset - SS2030},
  author={[Original Author]},
  year={2024},
  publisher={Kaggle},
  url={https://www.kaggle.com/datasets/...}
}
```

## License

The dataset license depends on the source. Please check:
- Kaggle dataset page for licensing information
- Original data provider's terms of use

## Issues or Questions?

If you encounter issues with the dataset:

1. Check the dataset is in the correct format (CSV with `text` and `Sentiment` columns)
2. Verify Arabic text encoding (UTF-8)
3. Ensure no missing values in required columns
4. See preprocessing logs in the notebook for details

For questions, please [open an issue](https://github.com/Bolaal/Arabic-Sentiment-Analysis-BiLSTM-vs-AraBERT/issues).

---

**Last Updated:** January 2025
```

---

## **📂 Final Data Folder Structure**

After creating these files:
```
data/
│
├── README.md              # ✅ Detailed documentation (content above)
│
├── raw/                   # Original downloaded data
│   └── .gitkeep          # ✅ Empty file (0 bytes)
│
└── processed/             # Preprocessed splits (generated by notebook)
    └── .gitkeep          # ✅ Empty file (0 bytes)