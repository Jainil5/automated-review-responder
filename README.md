# Automated Review Responder

An end-to-end NLP pipeline for **classifying customer clothing reviews** by intent and **generating automated, professional responses** using a fine-tuned language model.

---

## Project Overview

This project solves two problems:

1. **Intent Classification** — Given a customer review, classify it into one of 6 intent categories using multiple ML models.
2. **Response Generation** — Automatically generate a polite, professional company response to any review using a fine-tuned `flan-t5-base` model with LoRA adapters.

---

## Project Structure

```
automated-review-responder/
└── backend/
    ├── datasets/
    │   ├── clothing_reviews_intent.csv   # Labelled review dataset (intent classification)
    │   └── rewiews-response.csv          # Review–response pairs (responder fine-tuning)
    ├── models/
    │   └── flan_t5_review_lora/          # Saved LoRA fine-tuned model weights
    ├── results/
    │   ├── model_performance_summary.csv # All model metrics (aggregate + per-class)
    │   ├── *_confusion_matrix.csv        # Per-model confusion matrices
    │   └── charts/
    │       └── accuracy_comparison.png   # Model accuracy bar chart
    ├── services/
    │   ├── train_classifier.py           # Train & evaluate all classification models
    │   ├── train_responder.py            # Fine-tune flan-t5-base with LoRA
    │   └── main_responder.py             # Run inference with the fine-tuned model
    └── requirements.txt
```

---

## Part 1 — Intent Classifier

### Dataset

- **Source:** `clothing_reviews_intent.csv`
- **Total rows:** 4,000 (259 unique reviews after deduplication)
- **Classes (6):** `delivery`, `general`, `material`, `quality`, `refund`, `size_fit`
- Multi-label intent combos (e.g. `delivery|refund`) are resolved to the **primary intent** (first label).

### Preprocessing Pipeline

Each review is cleaned through the following steps before being fed to any model:

| Step | Description |
|------|-------------|
| Lowercase | Normalises case |
| URL / HTML removal | Strips `http://`, `<tags>`, etc. |
| Punctuation removal | Keeps only `a–z` and spaces |
| Whitespace normalisation | Collapses multiple spaces |
| Tokenisation | Word-level split via NLTK |
| Lemmatisation | `"looks"` → `"look"`, `"taking"` → `"take"` |

> **Note:** Stopword removal is intentionally skipped — reviews are 2–5 words long, and removing stopwords strips too much signal from short phrases.

### Models Evaluated

| Model | Notes |
|-------|-------|
| Naive Bayes | Multinomial NB on TF-IDF |
| Logistic Regression | `max_iter=1500`, TF-IDF (1–2 ngrams) |
| Random Forest | 200 estimators, TF-IDF |
| SVM | Linear kernel with probability calibration |
| Neural Network | Bidirectional LSTM + Embedding on tokenised sequences |

### Model Evaluation Results

> Test set: 52 samples (20% stratified split of 259 unique reviews)

| Model | Accuracy | Precision (weighted) | Recall (weighted) | F1 (weighted) | ROC-AUC |
|-------|:--------:|:--------------------:|:-----------------:|:-------------:|:-------:|
| Naive Bayes | 71.15% | 0.691 | 0.712 | 0.700 | 0.900 |
| Logistic Regression | 78.85% | 0.764 | 0.788 | 0.771 | 0.977 |
| Random Forest | 76.92% | 0.776 | 0.769 | 0.765 | 0.922 |
| **SVM** | **90.38%** | **0.927** | **0.904** | **0.903** | **0.997** |
| Neural Network | ~25–35% | — | — | — | — |

**SVM is the best-performing model** for this task. The Neural Network underperforms due to the very small dataset size (259 unique samples is insufficient for training deep weights reliably).

#### Per-Class Accuracy (SVM — Best Model)

| Class | Precision | Recall | F1 |
|-------|:---------:|:------:|:--:|
| delivery | 1.00 | 1.00 | 1.00 |
| general | 1.00 | 1.00 | 1.00 |
| material | 1.00 | 0.67 | 0.80 |
| quality | 1.00 | 0.82 | 0.90 |
| refund | 0.71 | 1.00 | 0.83 |
| size_fit | 0.92 | 1.00 | 0.96 |

Full per-class metrics for all models are saved in `backend/results/model_performance_summary.csv`.

---

## Part 2 — Response Generator

### Model

- **Base model:** `google/flan-t5-base` (encoder-decoder, seq2seq)
- **Fine-tuning method:** LoRA (Low-Rank Adaptation) via `peft`
  - Rank `r=8`, alpha `16`, dropout `0.1`
  - Target modules: `q`, `v` attention layers
- **Dataset:** `rewiews-response.csv` — review/response pairs
- **Training:** 10 epochs, batch size 2, gradient accumulation 4 steps, lr `2e-4`

### Inference

```python
from main_responder import generate_response

response = generate_response("The zipper on my jacket keeps getting stuck.")
print(response)
```

**Prompt template used:**
```
Respond professionally to this customer review on behalf of company.:
{review}
```

---

## Setup & Usage

### 1. Create virtual environment

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install torch peft nltk tensorflow
```

### 3. Run intent classifier training & evaluation

```bash
python backend/services/train_classifier.py
```

Outputs saved to `backend/results/`:
- `model_performance_summary.csv` — all metrics per model
- `*_confusion_matrix.csv` — confusion matrix per model
- `charts/accuracy_comparison.png` — accuracy bar chart

### 4. Fine-tune the response generator

```bash
python backend/services/train_responder.py
```

Saves the fine-tuned LoRA model to `backend/models/flan_t5_review_lora/`.

### 5. Run the response generator

```bash
python backend/services/main_responder.py
```

---

## Dependencies

```
matplotlib        # Charting
seaborn           # (optional) heatmaps
transformers      # flan-t5-base base model
peft              # LoRA fine-tuning
langchain         # (planned)
streamlit         # (planned frontend)
pandas            # Data handling
scikit-learn      # ML models & metrics
nltk              # Text preprocessing
tensorflow        # Neural network model
torch             # flan-t5 inference
```

---

## Key Design Decisions

- **Deduplication before split** — The raw dataset has 4,000 rows but only 259 unique texts. Without deduplication, the same review appears in both train and test, causing artificially inflated accuracy (96–100%). Deduplicating on `review` text before the split gives honest, generalisation-based scores.
- **Primary intent only** — Multi-label combos like `delivery|refund` are resolved to the first label (`delivery`) to keep classification simple and single-label.
- **No stopword removal** — Reviews average 4–5 words. Removing stopwords strips key signal tokens and significantly hurts accuracy on this short-text dataset.
- **LoRA over full fine-tuning** — Only ~0.1% of parameters are trainable, making it feasible to fine-tune `flan-t5-base` on CPU/MPS without large GPU resources.
