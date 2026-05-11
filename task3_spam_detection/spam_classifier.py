"""
SMS Spam Classifier ls ~/Desktop/"spam sms detection"

"""

import re
import string
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


# ---------- 1. Load & clean ----------
def load_data(path):
    df = pd.read_csv(path, encoding='latin-1')
    df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})
    df = df.dropna().drop_duplicates(subset=['message']).reset_index(drop=True)
    df['target'] = (df['label'] == 'spam').astype(int)
    return df


def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' URL ', text)
    text = re.sub(r'\b\d{5,}\b', ' NUM ', text)
    text = re.sub(r'[' + re.escape(string.punctuation) + r']', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


# ---------- 2. Train & evaluate ----------
def evaluate(name, pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    return {
        'name':      name,
        'pipeline':  pipe,
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall':    recall_score(y_test, y_pred),
        'f1':        f1_score(y_test, y_pred),
        'cm':        confusion_matrix(y_test, y_pred),
        'report':    classification_report(y_test, y_pred,
                                           target_names=['ham', 'spam'], digits=4),
    }


# ---------- 3. Plots ----------
def plot_class_distribution(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df['label'].value_counts()
    bars = ax.bar(counts.index, counts.values, color=['#4C9F70', '#E63946'])
    ax.set_title('Class Distribution', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of messages')
    for bar, c in zip(bars, counts.values):
        pct = c / len(df) * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                f'{c}\n({pct:.1f}%)', ha='center', fontsize=11)
    plt.tight_layout()
    plt.savefig('01_class_distribution.png', dpi=120, bbox_inches='tight')
    plt.close()


def plot_message_length(df):
    df = df.assign(length=df['message'].str.len())
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df[df['target'] == 0]['length'], bins=50, alpha=0.65,
            label='Ham', color='#4C9F70')
    ax.hist(df[df['target'] == 1]['length'], bins=50, alpha=0.65,
            label='Spam', color='#E63946')
    ax.set_title('Message Length Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Character count')
    ax.set_ylabel('Frequency')
    ax.set_xlim(0, 400)
    ax.legend()
    plt.tight_layout()
    plt.savefig('02_message_length.png', dpi=120, bbox_inches='tight')
    plt.close()


def plot_confusion_matrices(results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, r in zip(axes, results):
        sns.heatmap(r['cm'], annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'], ax=ax)
        ax.set_title(f"{r['name']}\nF1 = {r['f1']:.4f}", fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig('03_confusion_matrices.png', dpi=120, bbox_inches='tight')
    plt.close()


def plot_model_comparison(results):
    metrics_df = pd.DataFrame([{
        'Model':     r['name'],
        'Accuracy':  r['accuracy'],
        'Precision': r['precision'],
        'Recall':    r['recall'],
        'F1':        r['f1'],
    } for r in results]).set_index('Model')

    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_df.plot(kind='bar', ax=ax,
                    color=['#264653', '#2A9D8F', '#E9C46A', '#E76F51'])
    ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score')
    ax.set_ylim(0.85, 1.0)
    ax.set_xticklabels(metrics_df.index, rotation=0)
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=8, padding=2)
    plt.tight_layout()
    plt.savefig('04_model_comparison.png', dpi=120, bbox_inches='tight')
    plt.close()


def plot_top_words(lr_pipe):
    vectorizer = lr_pipe.named_steps['tfidf']
    clf = lr_pipe.named_steps['clf']
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = clf.coef_[0]

    top_spam = np.argsort(coefs)[-20:]
    top_ham  = np.argsort(coefs)[:20]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].barh(feature_names[top_spam], coefs[top_spam], color='#E63946')
    axes[0].set_title('Top 20 Spam-Indicative Words/Phrases', fontweight='bold')
    axes[0].set_xlabel('Coefficient (higher = more spammy)')

    axes[1].barh(feature_names[top_ham], coefs[top_ham], color='#4C9F70')
    axes[1].set_title('Top 20 Ham-Indicative Words/Phrases', fontweight='bold')
    axes[1].set_xlabel('Coefficient (lower = more hammy)')

    plt.tight_layout()
    plt.savefig('05_top_words.png', dpi=120, bbox_inches='tight')
    plt.close()


# ---------- 4. Main ----------
def main():
    df = load_data("spam.csv")
    df['clean'] = df['message'].apply(clean_text)

    print(f"Loaded {len(df)} unique messages "
          f"({df['target'].sum()} spam, {(df['target'] == 0).sum()} ham)\n")

    X_train, X_test, y_train, y_test = train_test_split(
        df['clean'], df['target'],
        test_size=0.2, random_state=42, stratify=df['target']
    )

    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95,
                            sublinear_tf=True, stop_words='english')

    classifiers = {
        'Naive Bayes':         MultinomialNB(alpha=0.1),
        'Logistic Regression': LogisticRegression(max_iter=1000,
                                                  class_weight='balanced',
                                                  C=4, random_state=42),
        'Linear SVM':          LinearSVC(C=1, class_weight='balanced',
                                         random_state=42),
    }

    results = []
    for name, clf in classifiers.items():
        pipe = Pipeline([('tfidf', tfidf), ('clf', clf)])
        pipe.fit(X_train, y_train)
        results.append(evaluate(name, pipe, X_test, y_test))

    # ---------- print metrics ----------
    print("=" * 72)
    print(f"{'Model':<24}{'Accuracy':>10}{'Precision':>12}{'Recall':>10}{'F1':>10}")
    print("=" * 72)
    for r in results:
        print(f"{r['name']:<24}{r['accuracy']:>10.4f}{r['precision']:>12.4f}"
              f"{r['recall']:>10.4f}{r['f1']:>10.4f}")
    print("=" * 72)

    for r in results:
        print(f"\n>> {r['name']}")
        print("Confusion matrix [rows=actual, cols=predicted]")
        print(f"             pred_ham  pred_spam")
        print(f"actual_ham  {r['cm'][0,0]:>9} {r['cm'][0,1]:>10}")
        print(f"actual_spam {r['cm'][1,0]:>9} {r['cm'][1,1]:>10}")
        print(r['report'])

    # ---------- generate plots ----------
    print("\nGenerating graphs...")
    plot_class_distribution(df)
    plot_message_length(df)
    plot_confusion_matrices(results)
    plot_model_comparison(results)
    lr_result = next(r for r in results if r['name'] == 'Logistic Regression')
    plot_top_words(lr_result['pipeline'])
    print("Saved 5 plots: 01_class_distribution.png, 02_message_length.png,")
    print("               03_confusion_matrices.png, 04_model_comparison.png,")
    print("               05_top_words.png")

    # ---------- save best model ----------
    best = max(results, key=lambda r: r['f1'])
    print(f"\nBest model by F1: {best['name']} (F1 = {best['f1']:.4f})")
    joblib.dump(best['pipeline'], 'spam_classifier.joblib')
    print("Saved best pipeline -> spam_classifier.joblib")

    # ---------- demo predictions ----------
    samples = [
        "Hey, are we still meeting for lunch tomorrow at 1pm?",
        "WINNER!! You have been selected to receive a 1000 dollar prize. Call 09061701461 now!",
        "Don't forget to pick up milk on the way home.",
        "URGENT! Your account has been suspended. Click http://bit.ly/abc123 to verify.",
        "Free entry in 2 a wkly comp to win FA Cup final tkts. Text FA to 87121.",
        "Can you send me the notes from yesterday's class?",
    ]    
    cleaned = [clean_text(s) for s in samples]
    preds = best['pipeline'].predict(cleaned)
    print(f"\n>> Sample predictions using {best['name']}:")
    for msg, p in zip(samples, preds):
        label = "SPAM" if p == 1 else "HAM "
        print(f"  [{label}] {msg[:70]}{'...' if len(msg) > 70 else ''}")


if __name__ == "__main__":
    main()
