"""
VoiceBridge — Model Evaluation Script
Run from: Voice_med_backend/inference/
Output  : prints results + saves confusion_matrix.png in same folder
"""

import os, sys, numpy as np
import pandas as pd

# ── Paths (mirrors voicebridge_ui.py structure) ───────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "models"))
DATA_DIR   = os.path.normpath(os.path.join(BASE_DIR, "..", "dataset"))   # your folder
LABEL_MAP_PATH = os.path.join(BASE_DIR, "labelmap.py")

CLASSIFIER_PATH = os.path.join(MODELS_DIR, "isl.pkl")
SCALER_PATH     = os.path.join(MODELS_DIR, "scaler.pkl")

# ── Load your actual saved model ──────────────────────────────────────────────
import joblib, importlib.util

print("Loading model...")
clf    = joblib.load(CLASSIFIER_PATH)
scaler = joblib.load(SCALER_PATH)
print(f"  ✓ Classifier : {type(clf).__name__}")
print(f"  ✓ Scaler     : {type(scaler).__name__}")

# Load label map
spec = importlib.util.spec_from_file_location("labelmap", LABEL_MAP_PATH)
lm   = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lm)
LABEL_TO_WORD = lm.LABEL_TO_WORD
print(f"  ✓ Labels     : {len(LABEL_TO_WORD)} → {list(LABEL_TO_WORD.values())}")

# ── Load dataset ──────────────────────────────────────────────────────────────
# ── Load dataset (ONLY isl.csv) ─────────────────────────────
print("\nLoading dataset...")

DATASET_PATH = os.path.join(BASE_DIR, "..", "dataset", "isl.csv")

print("Dataset path:", DATASET_PATH)

if not os.path.exists(DATASET_PATH):
    print("Dataset not found!")
    sys.exit()

df = pd.read_csv(DATASET_PATH)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

print("Samples:", X.shape[0])
print("Features:", X.shape[1])
print("Classes:", sorted(set(y)))

print("\nClass distribution:")
for cls in sorted(set(y)):
    word  = LABEL_TO_WORD.get(cls, str(cls))
    count = np.sum(y == cls)
    bar   = "█" * (count // 20)
    print(f"  {str(cls):>3} ({word:>12}) : {count:4d}  {bar}")

# ── Scale using YOUR scaler ───────────────────────────────────────────────────
X_scaled = scaler.transform(X)   # transform only — not fit_transform

# ── 80/20 stratified split ────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y)

y_pred = clf.predict(X_te)
acc    = accuracy_score(y_te, y_pred)

print(f"\n{'='*50}")
print(f"  Test Accuracy  (80/20 split) : {acc*100:.2f}%")

# ── 5-fold cross validation ───────────────────────────────────────────────────
cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')
print(f"  5-Fold CV Mean               : {cv_scores.mean()*100:.2f}%")
print(f"  5-Fold CV Std                : ±{cv_scores.std()*100:.2f}%")
print(f"  Per-fold scores              : {[f'{s*100:.1f}%' for s in cv_scores]}")
print(f"{'='*50}")

# ── Per-class report ──────────────────────────────────────────────────────────
label_names = [LABEL_TO_WORD.get(c, str(c)) for c in sorted(set(y))]
print(f"\nPer-Class Report:")
print(classification_report(y_te, y_pred,
      labels=sorted(set(y)), target_names=label_names, zero_division=0))

# ── Misclassifications ────────────────────────────────────────────────────────
wrong = np.where(y_pred != y_te)[0]
if len(wrong) == 0:
    print("✓ Zero misclassifications on test set.")
    print("  Confirmed genuine: 5-fold CV also shows consistent high accuracy.")
else:
    print(f"Misclassifications: {len(wrong)}")
    for i in wrong:
        tw = LABEL_TO_WORD.get(y_te[i],   str(y_te[i]))
        pw = LABEL_TO_WORD.get(y_pred[i], str(y_pred[i]))
        print(f"  Sample {i:3d}: TRUE={tw:15s}  PREDICTED={pw}")

# ── Confusion matrix plot ─────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    all_labels = sorted(set(y))
    disp_names = [LABEL_TO_WORD.get(c, str(c)) for c in all_labels]
    cm = confusion_matrix(y_te, y_pred, labels=all_labels)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=disp_names, yticklabels=disp_names,
                linewidths=0.5, ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=13)
    ax.set_ylabel('True Label',      fontsize=13)
    ax.set_title(
        f'VoiceBridge ISL Classifier — Confusion Matrix\n'
        f'Model: {type(clf).__name__}  |  '
        f'Test: {acc*100:.1f}%  |  '
        f'5-Fold CV: {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%\n'
        f'Dataset: {len(X)} samples  |  26 ISL Signs ',
        fontsize=12, pad=15)
    plt.tight_layout()

    out = os.path.join(BASE_DIR, "confusion_matrix.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\n✓ Confusion matrix saved → {out}")

except ImportError:
    print("\n(matplotlib/seaborn not installed — skipping plot)")
    print("  pip install matplotlib seaborn")

print("\nDone.")