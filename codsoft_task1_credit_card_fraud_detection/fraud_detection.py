"""
Credit Card Fraud Detection


"""

import warnings
warnings.filterwarnings("ignore")

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_score, recall_score, f1_score, accuracy_score,
)


# -------------------------------------------------------------------
# 1. Load
# -------------------------------------------------------------------
TRAIN_PATH = "fraudTrain.csv"
TEST_PATH  = "fraudTest.csv"
OUT_DIR    = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading data...")
t0 = time.time()
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)
print(f"  Train: {train_df.shape}, Test: {test_df.shape}  [{time.time()-t0:.1f}s]")
print(f"  Fraud rate -- train: {train_df['is_fraud'].mean()*100:.3f}%  "
      f"test: {test_df['is_fraud'].mean()*100:.3f}%")


# -------------------------------------------------------------------
# 2. Feature engineering
# -------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between cardholder and merchant in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def engineer(df):
    df = df.copy()
    # Time features
    ts = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"]       = ts.dt.hour
    df["day"]        = ts.dt.day
    df["dayofweek"]  = ts.dt.dayofweek
    df["month"]      = ts.dt.month

    # Age at time of transaction
    dob = pd.to_datetime(df["dob"])
    df["age"] = ((ts - dob).dt.days / 365.25).astype(int)

    # Distance from cardholder home to merchant
    df["distance_km"] = haversine_km(df["lat"], df["long"],
                                     df["merch_lat"], df["merch_long"])
    return df


print("Feature engineering...")
train_df = engineer(train_df)
test_df  = engineer(test_df)


# -------------------------------------------------------------------
# 3. Select features, encode categoricals
# -------------------------------------------------------------------
# Drop PII, identifiers, high-cardinality columns, and raw datetime
DROP_COLS = [
    "Unnamed: 0", "trans_date_trans_time", "cc_num", "first", "last",
    "street", "city", "zip", "trans_num", "unix_time", "dob",
]

NUMERIC = ["amt", "lat", "long", "city_pop", "merch_lat", "merch_long",
           "hour", "day", "dayofweek", "month", "age", "distance_km"]

ONEHOT  = ["category", "gender"]                  # low cardinality
FREQENC = ["merchant", "job", "state"]            # higher cardinality


def freq_encode(train, test, cols):
    """Replace category with its frequency in the training set."""
    for c in cols:
        freq = train[c].value_counts(normalize=True)
        train[c + "_freq"] = train[c].map(freq).fillna(0.0)
        test[c  + "_freq"] = test[c].map(freq).fillna(0.0)
    return train, test


train_df, test_df = freq_encode(train_df, test_df, FREQENC)

# One-hot encode low-cardinality columns (fit on combined to keep columns aligned)
combined = pd.concat([train_df[ONEHOT], test_df[ONEHOT]], axis=0)
combined = pd.get_dummies(combined, columns=ONEHOT, drop_first=True)
train_oh = combined.iloc[:len(train_df)].reset_index(drop=True)
test_oh  = combined.iloc[len(train_df):].reset_index(drop=True)

freq_cols = [c + "_freq" for c in FREQENC]
X_train = pd.concat([train_df[NUMERIC + freq_cols].reset_index(drop=True), train_oh], axis=1).astype(np.float32)
X_test  = pd.concat([test_df[NUMERIC  + freq_cols].reset_index(drop=True), test_oh],  axis=1).astype(np.float32)
y_train = train_df["is_fraud"].values
y_test  = test_df["is_fraud"].values

# Stratified subsample of training set to fit in memory while preserving fraud ratio.
# Keeps all fraud cases + a random sample of legit cases.
SAMPLE_LEGIT = 400_000
rng = np.random.RandomState(42)
fraud_idx = np.where(y_train == 1)[0]
legit_idx = np.where(y_train == 0)[0]
legit_sample = rng.choice(legit_idx, size=min(SAMPLE_LEGIT, len(legit_idx)), replace=False)
keep = np.concatenate([fraud_idx, legit_sample])
rng.shuffle(keep)
X_train = X_train.iloc[keep].reset_index(drop=True)
y_train = y_train[keep]

# Free the original DataFrames
del train_df, combined, train_oh, test_oh
import gc; gc.collect()

print(f"  Final feature matrix: train {X_train.shape}, test {X_test.shape}")
print(f"  Train fraud rate after subsample: {y_train.mean()*100:.3f}%")


# -------------------------------------------------------------------
# 4. Train models
# -------------------------------------------------------------------
# Scale features for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000, class_weight="balanced",
                           solver="liblinear", random_state=42),
        X_train_scaled, X_test_scaled,
    ),
    "Decision Tree": (
        DecisionTreeClassifier(max_depth=15, class_weight="balanced",
                               random_state=42),
        X_train.values, X_test.values,
    ),
    "Random Forest": (
        RandomForestClassifier(n_estimators=100, max_depth=20,
                               class_weight="balanced", n_jobs=-1,
                               random_state=42),
        X_train.values, X_test.values,
    ),
}

results = {}
for name, (model, Xtr, Xte) in models.items():
    print(f"\nTraining {name}...")
    t0 = time.time()
    model.fit(Xtr, y_train)
    train_time = time.time() - t0

    y_pred  = model.predict(Xte)
    y_proba = model.predict_proba(Xte)[:, 1]

    results[name] = {
        "model":     model,
        "y_pred":    y_pred,
        "y_proba":   y_proba,
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_proba),
        "cm":        confusion_matrix(y_test, y_pred),
        "train_time": train_time,
    }
    r = results[name]
    print(f"  {name} trained in {train_time:.1f}s")
    print(f"  Accuracy : {r['accuracy']:.4f}")
    print(f"  Precision: {r['precision']:.4f}")
    print(f"  Recall   : {r['recall']:.4f}")
    print(f"  F1       : {r['f1']:.4f}")
    print(f"  ROC-AUC  : {r['roc_auc']:.4f}")


# -------------------------------------------------------------------
# 5. Comparison table + classification reports
# -------------------------------------------------------------------
print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)
summary = pd.DataFrame({
    name: {
        "Accuracy":  r["accuracy"],
        "Precision": r["precision"],
        "Recall":    r["recall"],
        "F1":        r["f1"],
        "ROC-AUC":   r["roc_auc"],
        "Train (s)": r["train_time"],
    }
    for name, r in results.items()
}).T
print(summary.round(4).to_string())
summary.round(4).to_csv(os.path.join(OUT_DIR, "model_comparison.csv"))

print("\nClassification reports:")
for name, r in results.items():
    print(f"\n--- {name} ---")
    print(classification_report(y_test, r["y_pred"],
                                target_names=["Legit", "Fraud"], digits=4))


# -------------------------------------------------------------------
# 6. Plots
# -------------------------------------------------------------------
print("Generating plots...")

# 6a. Class distribution
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, y, title in [(axes[0], y_train, "Train (subsampled)"),
                     (axes[1], y_test,  "Test")]:
    counts = pd.Series(y).value_counts().sort_index()
    ax.bar(["Legit", "Fraud"], counts.values, color=["#4C72B0", "#C44E52"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n({v/counts.sum()*100:.2f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.set_title(f"{title} class distribution")
    ax.set_ylabel("Count")
    ax.set_yscale("log")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "01_class_distribution.png"), dpi=120)
plt.close()

# 6b. Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (name, r) in zip(axes, results.items()):
    cm = r["cm"]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(name)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Legit", "Fraud"]); ax.set_yticklabels(["Legit", "Fraud"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black",
                    fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "02_confusion_matrices.png"), dpi=120)
plt.close()

# 6c. ROC curves
plt.figure(figsize=(7, 6))
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {r['roc_auc']:.4f})", linewidth=2)
plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "03_roc_curves.png"), dpi=120)
plt.close()

# 6d. Feature importance (Random Forest)
rf = results["Random Forest"]["model"]
importance = pd.Series(rf.feature_importances_, index=X_train.columns)
top = importance.sort_values(ascending=True).tail(15)
plt.figure(figsize=(8, 6))
plt.barh(top.index, top.values, color="#4C72B0")
plt.xlabel("Importance")
plt.title("Random Forest — Top 15 Feature Importances")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "04_feature_importance.png"), dpi=120)
plt.close()

# 6e. Metric comparison bar chart
metrics = ["Precision", "Recall", "F1", "ROC-AUC"]
x = np.arange(len(metrics))
width = 0.25
plt.figure(figsize=(9, 5))
colors = ["#4C72B0", "#DD8452", "#55A868"]
for i, (name, color) in enumerate(zip(results.keys(), colors)):
    vals = [results[name]["precision"], results[name]["recall"],
            results[name]["f1"], results[name]["roc_auc"]]
    plt.bar(x + i*width, vals, width, label=name, color=color)
    for j, v in enumerate(vals):
        plt.text(x[j] + i*width, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
plt.xticks(x + width, metrics)
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("Model Comparison")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "05_metric_comparison.png"), dpi=120)
plt.close()

print(f"\nDone. Outputs saved to {OUT_DIR}/")
print("Files:")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"  {f}")
