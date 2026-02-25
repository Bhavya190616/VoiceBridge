import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------
# Paths
# -----------------------------
DATASET_PATH = os.path.join("dataset", "isl_landmarks.csv")
MODEL_PATH = os.path.join("models", "isl_classifier.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")

# -----------------------------
# Load dataset
# -----------------------------
data = pd.read_csv(DATASET_PATH)

X = data.iloc[:, :-1].values   # 63 landmark features
y = data.iloc[:, -1].values   # labels

print("Dataset loaded")
print("Samples:", X.shape[0])
print("Features:", X.shape[1])
print("Classes:", np.unique(y))

# -----------------------------
# Train / test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# -----------------------------
# Feature normalization
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Train SVM
# -----------------------------
svm = SVC(kernel="rbf", C=10, gamma="scale")
svm.fit(X_train, y_train)

# -----------------------------
# Evaluation
# -----------------------------
y_pred = svm.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# Save model and scaler
# -----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(svm, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("\nModel saved to:", MODEL_PATH)
print("Scaler saved to:", SCALER_PATH)
