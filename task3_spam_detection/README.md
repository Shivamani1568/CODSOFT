# Task 3: SMS Spam Detection

This is my third task for the CodSoft Machine Learning Internship. The goal was to build a model that can classify SMS messages as either spam or legitimate (ham), using TF-IDF features and classical ML classifiers.

## Dataset

I used the UCI SMS Spam Collection dataset, which has 5,572 labeled SMS messages. After dropping duplicates I ended up with 5,169 unique messages — 4,516 ham and 653 spam. So the classes are imbalanced (about 87% ham, 13% spam), which is something I had to keep in mind when picking the right metric.

## Approach

I tried three classifiers on the same TF-IDF features and compared them:

- Multinomial Naive Bayes
- Logistic Regression (with class_weight='balanced' to handle the imbalance)
- Linear SVM (also with class_weight='balanced')

For the features I used TF-IDF with unigrams and bigrams, English stopwords removed, sublinear TF scaling, and min_df=2 to drop very rare terms.

Before vectorizing I cleaned the text: lowercased everything, replaced URLs with a `URL` placeholder, replaced long digit runs (5+ digits) with a `NUM` placeholder, and stripped punctuation. The placeholder trick helps because spam messages tend to have phone numbers and shortcodes, and I wanted the model to learn from the pattern, not from specific numbers.

I split the data 80/20 with stratification so both train and test had the same class ratio.

## Results

On the held-out test set (1,034 messages):

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Naive Bayes | 0.9836 | 0.9750 | 0.8931 | 0.9323 |
| Logistic Regression | 0.9865 | 0.9606 | 0.9313 | 0.9457 |
| Linear SVM | 0.9874 | 0.9758 | 0.9237 | 0.9490 |

Linear SVM came out on top by F1 score, catching 121 out of 131 spam messages with only 3 false positives on ham. I focused on F1 on the spam class because accuracy alone is misleading on imbalanced data — a model that just predicts "ham" for everything would already get around 87% accuracy.

Naive Bayes was the most conservative — highest precision but lowest recall (missed 14 spam messages). Logistic Regression caught the most spam but also flagged 5 legitimate messages as spam. Linear SVM was the best balance.

## What the model learned

The top spam-indicative words from the Logistic Regression coefficients were what I'd expect: `free`, `txt`, `claim`, `won`, `call`, `prize`, `urgent`, `mobile`, plus my `NUM` placeholder (which makes sense — long digit sequences almost always show up in spam). The top ham-indicative words were everyday casual stuff like `ok`, `ll`, `gt`, `lt`, `sorry`, `going`, `later`.

## Files

- `spam_classifier.py` — full training and evaluation script
- `spam_classifier.joblib` — the saved Linear SVM pipeline (TF-IDF + classifier together)
- `01_class_distribution.png` — bar chart showing the ham/spam imbalance
- `02_message_length.png` — histogram of message lengths by class
- `03_confusion_matrices.png` — side-by-side heatmaps for all three models
- `04_model_comparison.png` — grouped bar chart of accuracy/precision/recall/F1
- `05_top_words.png` — the most spam-indicative and ham-indicative words

## How to run
pip install -r requirements.txt
python spam_classifier.py

This trains all three models, prints the metrics, saves the best one as `spam_classifier.joblib`, and generates the 5 plots.

To use the saved model on a new message:

```python
import joblib
model = joblib.load('spam_classifier.joblib')
print(model.predict(["WINNER! Click here to claim your prize"]))  # -> [1]
```

## What I learned

The biggest takeaway was that on imbalanced data, looking at accuracy alone hides a lot. The confusion matrices made it really clear which errors each model was making — Naive Bayes missing spam vs Logistic Regression over-flagging ham. Also, just plotting the top coefficients from Logistic Regression turned out to be a great sanity check — when the model says "free" and "claim" are spammy, you know it's learning the right thing and not some weird artifact.

The `URL` and `NUM` placeholder trick was something I added after looking at the raw messages — spam messages had tons of unique phone numbers and shortcodes, and without the placeholders the model would have wasted features on memorizing specific numbers.