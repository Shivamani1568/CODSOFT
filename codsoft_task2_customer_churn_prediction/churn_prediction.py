"""
Customer Churn Prediction
Dataset: Bank Customer Churn (10,000 rows)
Models: Logistic Regression, Random Forest, Gradient Boosting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
import warnings
warnings.filterwarnings("ignore")

# ---------- 1. LOAD DATA ----------
df = pd.read_csv("Churn_Modelling.csv")
print(f"Shape: {df.shape}")
print(f"\nChurn distribution:\n{df['Exited'].value_counts()}")
print(f"Churn rate: {df['Exited'].mean()*100:.2f}%")
print(f"\nMissing values: {df.isnull().sum().sum()}")

# ---------- 2. PREPROCESS ----------
# Drop identifier columns (no predictive value)
df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

X = df.drop(columns=["Exited"])
y = df["Exited"]

categorical = ["Geography", "Gender"]
numeric = ["CreditScore", "Age", "Tenure", "Balance",
           "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(drop="first"), categorical),
])

# Stratified split to preserve churn ratio in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

# ---------- 3. TRAIN MODELS ----------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42),
}

results = {}
trained = {}

for name, clf in models.items():
    pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    results[name] = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall":    recall_score(y_test, y_pred),
        "F1":        f1_score(y_test, y_pred),
        "ROC-AUC":   roc_auc_score(y_test, y_prob),
    }
    trained[name] = pipe

# ---------- 4. RESULTS ----------
results_df = pd.DataFrame(results).T.round(4)
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)
print(results_df)

best_model_name = results_df["ROC-AUC"].idxmax()
best_model = trained[best_model_name]
print(f"\nBest model (by ROC-AUC): {best_model_name}")

print(f"\n--- Detailed report for {best_model_name} ---")
print(classification_report(y_test, best_model.predict(X_test),
                            target_names=["Stayed", "Churned"]))

# ---------- 5. VISUALIZATIONS ----------
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# (a) Metrics comparison bar chart
results_df.plot(kind="bar", ax=axes[0, 0], colormap="viridis")
axes[0, 0].set_title("Model Performance Comparison", fontweight="bold")
axes[0, 0].set_ylabel("Score")
axes[0, 0].set_xticklabels(results_df.index, rotation=15)
axes[0, 0].legend(loc="lower right", fontsize=8)
axes[0, 0].set_ylim(0, 1)
axes[0, 0].grid(axis="y", alpha=0.3)

# (b) ROC curves
for name, pipe in trained.items():
    y_prob = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    axes[0, 1].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
axes[0, 1].plot([0, 1], [0, 1], "k--", alpha=0.4)
axes[0, 1].set_title("ROC Curves", fontweight="bold")
axes[0, 1].set_xlabel("False Positive Rate")
axes[0, 1].set_ylabel("True Positive Rate")
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# (c) Confusion matrix for best model
cm = confusion_matrix(y_test, best_model.predict(X_test))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1, 0],
            xticklabels=["Stayed", "Churned"],
            yticklabels=["Stayed", "Churned"])
axes[1, 0].set_title(f"Confusion Matrix — {best_model_name}", fontweight="bold")
axes[1, 0].set_xlabel("Predicted")
axes[1, 0].set_ylabel("Actual")

# (d) Feature importance from the best tree-based model (or coefficients for LR)
feat_names = numeric + ["Geography_Germany", "Geography_Spain", "Gender_Male"]
clf = best_model.named_steps["clf"]
if hasattr(clf, "feature_importances_"):
    importances = clf.feature_importances_
    title = f"Feature Importance — {best_model_name}"
else:
    importances = np.abs(clf.coef_[0])
    title = f"Feature |Coefficients| — {best_model_name}"

imp_df = pd.DataFrame({"feature": feat_names, "importance": importances}) \
            .sort_values("importance", ascending=True)
axes[1, 1].barh(imp_df["feature"], imp_df["importance"], color="teal")
axes[1, 1].set_title(title, fontweight="bold")
axes[1, 1].set_xlabel("Importance")

plt.tight_layout()
plt.savefig("churn_results.png", dpi=110, bbox_inches="tight")
plt.close()
print("\nSaved plots → churn_results.png")

# ---------- 6. SAMPLE PREDICTIONS ----------
print("\n--- Sample predictions on test set ---")
sample = X_test.head(5).copy()
sample["Actual"] = y_test.head(5).values
sample["Predicted"] = best_model.predict(X_test.head(5))
sample["Churn_Probability"] = best_model.predict_proba(X_test.head(5))[:, 1].round(3)
print(sample[["Age", "Geography", "Balance", "NumOfProducts",
              "IsActiveMember", "Actual", "Predicted", "Churn_Probability"]])
