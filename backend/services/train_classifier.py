import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('wordnet',   quiet=True)
nltk.download('punkt_tab', quiet=True)

_lemmatizer = WordNetLemmatizer()

def preprocess(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'http\S+|www\S+|<[^>]+>', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [_lemmatizer.lemmatize(t) for t in tokens if t.strip()]
    return ' '.join(tokens)

# ──────────────────────────────────────────────
# 1. Load & Prepare Dataset
# ──────────────────────────────────────────────
df = pd.read_csv('backend/datasets/clothing_reviews_intent.csv')
df = df.dropna()

# Resolve multi-label combos → primary intent (first label before '|')
df['intent'] = df['intent'].str.split('|').str[0].str.strip()

# ── De-duplicate on review text to prevent data leakage ──
# The dataset has 4000 rows but only ~259 unique review strings.
# Without deduplication the same text lands in both train & test,
# causing artificially inflated (near-perfect) accuracy.
df_unique = df.drop_duplicates(subset='review').reset_index(drop=True)

# ── Apply preprocessing pipeline ─────────────
print("Preprocessing text...")
df_unique['review_clean'] = df_unique['review'].apply(preprocess)

X     = df_unique['review_clean']
y_raw = df_unique['intent']

print(f"Classes ({y_raw.nunique()}): {sorted(y_raw.unique())}")
print(f"Unique reviews after dedup: {len(df_unique)} (was {len(df)} with duplicates)")
print(f"Sample preprocessed: '{df_unique['review'].iloc[0]}' → '{df_unique['review_clean'].iloc[0]}'\n")

# Label encode
le        = LabelEncoder()
y_encoded = le.fit_transform(y_raw)
classes   = le.classes_
n_classes = len(classes)

# ──────────────────────────────────────────────
# 2. Train / Test Split (stratified)
# ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ──────────────────────────────────────────────
# 3. TF-IDF Vectorisation
# ──────────────────────────────────────────────
tfidf         = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

# Binarise test labels for ROC-AUC
y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))

# ──────────────────────────────────────────────
# 4. Output directory
# ──────────────────────────────────────────────
OUT_DIR   = 'backend/results'
CHART_DIR = os.path.join(OUT_DIR, 'charts')
os.makedirs(CHART_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# 5. Metrics helper
# ──────────────────────────────────────────────
def compute_metrics(name, y_true, y_pred, y_proba=None):
    acc      = accuracy_score(y_true, y_pred)
    prec_w   = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec_w    = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_w     = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    prec_mac = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_mac  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_mac   = f1_score(y_true, y_pred, average='macro', zero_division=0)

    roc_auc = None
    if y_proba is not None and n_classes > 1:
        try:
            roc_auc = roc_auc_score(
                y_test_bin, y_proba, average='weighted', multi_class='ovr'
            )
        except ValueError:
            roc_auc = None

    row = {
        'Model':                  name,
        'Accuracy':               round(acc,      4),
        'Precision (weighted)':   round(prec_w,   4),
        'Recall (weighted)':      round(rec_w,    4),
        'F1 (weighted)':          round(f1_w,     4),
        'Precision (macro)':      round(prec_mac, 4),
        'Recall (macro)':         round(rec_mac,  4),
        'F1 (macro)':             round(f1_mac,   4),
        'ROC-AUC (OvR weighted)': round(roc_auc, 4) if roc_auc else 'N/A',
    }

    # Per-class precision, recall, F1
    prec_per  = precision_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(n_classes)))
    rec_per   = recall_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(n_classes)))
    f1_per    = f1_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(n_classes)))
    support   = np.bincount(y_true, minlength=n_classes)

    for cls, p, r, f, s in zip(classes, prec_per, rec_per, f1_per, support):
        row[f'{cls}_precision'] = round(p, 4)
        row[f'{cls}_recall']    = round(r, 4)
        row[f'{cls}_f1']        = round(f, 4)
        row[f'{cls}_support']   = int(s)

    return row

def save_confusion_matrix(name, y_true, y_pred):
    cm    = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    fname = name.lower().replace(' ', '_') + '_confusion_matrix.csv'
    cm_df.to_csv(os.path.join(OUT_DIR, fname))
    print(f"  ✓ Confusion matrix → {fname}")

# ──────────────────────────────────────────────
# 6. Sklearn Models
# ──────────────────────────────────────────────
sklearn_models = {
    'Naive Bayes':         MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1500),
    'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42),
    'SVM':                 SVC(kernel='linear', probability=True),
}

all_metrics = []

for name, model in sklearn_models.items():
    print(f"Training {name}...")
    model.fit(X_train_tfidf, y_train)
    preds  = model.predict(X_test_tfidf)
    probas = model.predict_proba(X_test_tfidf) if hasattr(model, 'predict_proba') else None

    metrics = compute_metrics(name, y_test, preds, probas)
    all_metrics.append(metrics)

    print(f"  Accuracy: {metrics['Accuracy']*100:.2f}%")
    print(classification_report(y_test, preds, target_names=classes, zero_division=0))
    save_confusion_matrix(name, y_test, preds)

# ──────────────────────────────────────────────
# 7. Neural Network
# ──────────────────────────────────────────────
print("\nTraining Neural Network...")
MAXLEN = 30

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAXLEN)
X_test_seq  = pad_sequences(tokenizer.texts_to_sequences(X_test),  maxlen=MAXLEN)

# One-hot encode (aligned on all classes)
y_train_ohe = tf.keras.utils.to_categorical(y_train, num_classes=n_classes)
y_test_ohe  = tf.keras.utils.to_categorical(y_test,  num_classes=n_classes)

nn = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(MAXLEN,)),
    tf.keras.layers.Embedding(5000, 32),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(n_classes, activation='softmax')
])
nn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
nn.fit(X_train_seq, y_train_ohe, epochs=50, batch_size=32,
       validation_split=0.1, verbose=0)

nn_probas = nn.predict(X_test_seq, verbose=0)
nn_preds  = np.argmax(nn_probas, axis=1)

metrics = compute_metrics('Neural Network', y_test, nn_preds, nn_probas)
all_metrics.append(metrics)
print(f"  Accuracy: {metrics['Accuracy']*100:.2f}%")
print(classification_report(y_test, nn_preds, target_names=classes, zero_division=0))
save_confusion_matrix('Neural Network', y_test, nn_preds)

# ──────────────────────────────────────────────
# 8. Save Summary CSV
# ──────────────────────────────────────────────
summary_df   = pd.DataFrame(all_metrics)
summary_path = os.path.join(OUT_DIR, 'model_performance_summary.csv')
summary_df.to_csv(summary_path, index=False)

print("\n" + "="*60)
print("             MODEL PERFORMANCE SUMMARY")
print("="*60)
aggregate_cols = ['Model', 'Accuracy', 'Precision (weighted)', 'Recall (weighted)',
                  'F1 (weighted)', 'Precision (macro)', 'Recall (macro)',
                  'F1 (macro)', 'ROC-AUC (OvR weighted)']
print(summary_df[aggregate_cols].to_string(index=False))
print(f"\n✓ Full metrics CSV (aggregate + per-class) → {summary_path}")
PALETTE  = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']
BG       = '#F8F9FA'
GRID_COL = '#DEE2E6'

plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'axes.spines.top':  False,
    'axes.spines.right': False,
    'axes.facecolor':   BG,
    'figure.facecolor': 'white',
    'axes.grid':        True,
    'grid.color':       GRID_COL,
    'grid.linewidth':   0.7,
})

model_names = summary_df['Model'].tolist()
vals        = summary_df['Accuracy'].tolist()
colors      = [PALETTE[i % len(PALETTE)] for i in range(len(model_names))]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(model_names, vals, color=colors, edgecolor='white', height=0.5)
for bar, v in zip(bars, vals):
    ax.text(v + 0.008, bar.get_y() + bar.get_height() / 2,
            f'{v*100:.2f}%', va='center', fontsize=10, fontweight='bold')
ax.set_xlim(0, 1.15)
ax.set_xlabel('Accuracy')
ax.set_title('Model Accuracy Comparison', fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, 'accuracy_comparison.png'), dpi=150)
plt.close()

print(f"\n✓ Chart saved  → {CHART_DIR}/accuracy_comparison.png")
print(f"✓ Metrics CSV  → {summary_path}")
print(f"✓ Confusion matrices → {OUT_DIR}/<model>_confusion_matrix.csv")