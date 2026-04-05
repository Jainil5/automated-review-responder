# Automated Review Responder

An end-to-end NLP pipeline for **classifying customer clothing reviews** by intent and **generating automated, professional responses** using a fine-tuned language model.

---

## Project Overview

This project solves two problems:

1. **Intent Classification** — Given a customer review, classify it into one of 6 intent categories using multiple ML models.
2. **Response Generation** — Automatically generate a polite, professional company response to any review using a fine-tuned `flan-t5-base` model with LoRA adapters.

---

## Part 1 — Intent Classifier

### Dataset

- **Source:** `clothing_reviews_intent.csv`
- **Total rows:** 4,000 (259 unique reviews after deduplication)
- **Classes (6):** `delivery`, `general`, `material`, `quality`, `refund`, `size_fit`


### Model Evaluation Results

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

