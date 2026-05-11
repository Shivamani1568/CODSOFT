# Task 1: Credit Card Fraud Detection

Built a model to flag fraudulent credit card transactions. The dataset is from Kaggle (Sparkov-generated transactions) — about 1.3 million transactions for training and 555k for testing.

The main challenge here is class imbalance. Only 0.58% of transactions are actually fraud, so a model that just predicts "not fraud" for everything would get 99.4% accuracy and be completely useless. I used precision, recall, F1, and ROC-AUC instead of just accuracy.

## What I did

**Feature engineering.** The raw data has a timestamp and date of birth, so I pulled out hour, day, day-of-week, and month from the transaction time, and calculated the cardholder's age. I also computed the haversine distance between the cardholder's home and the merchant — the idea is that fraud often happens far from where the cardholder normally lives.

**Encoding.** Category and gender are low-cardinality so I one-hot encoded them. Merchant, job, and state have way too many unique values for one-hot, so I used frequency encoding instead. Dropped all the PII (credit card number, name, street, etc.) along with the raw timestamp.

**Models.** Tried three:
- Logistic Regression
- Decision Tree (max_depth=15)
- Random Forest (100 trees, max_depth=20)

All three use `class_weight='balanced'` so the model actually pays attention to the rare fraud class.

## Results

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.034 | 0.736 | 0.066 | 0.909 |
| Decision Tree | 0.216 | 0.940 | 0.352 | 0.971 |
| Random Forest | **0.666** | **0.827** | **0.738** | **0.994** |

Random Forest won by a lot. On the test set it caught 1,771 out of 2,145 fraud cases with only 870 false alarms across 555k transactions.

Logistic Regression has decent recall but its precision is awful — too many false positives — so it's not really useful. Decision Tree was a big jump but still flagged too many legit transactions as fraud. Random Forest gives the best balance.

## Why Random Forest works best here

It captures non-linear interactions between features. Fraud isn't "amount > $500" or "hour < 5am" — it's combinations like "high amount AND late night AND far from home AND unfamiliar merchant". A single tree can find some of those splits but a forest of 100 trees averages them out and overfits less.

## Most important features

Looking at the feature importance from Random Forest:
1. Transaction amount (~52%)
2. Hour of day (~18%)
3. Merchant frequency (~4%)

So basically: how much, when, and where. Makes sense.

## How to run it

Get the data from Kaggle: https://www.kaggle.com/datasets/kartik2112/fraud-detection — you need `fraudTrain.csv` and `fraudTest.csv`. Drop them in this folder.

Takes about 20-30 seconds on an M-series Mac. The plots and a comparison CSV will be saved to `outputs/`.

## Things I'd try next

- XGBoost or LightGBM — usually beats Random Forest on tabular data
- SMOTE for oversampling the fraud class instead of just using class weights
- Threshold tuning — the default 0.5 isn't necessarily right, lowering it would catch more fraud at the cost of more false alarms
- Per-card behavioral features like "average transaction amount for this card in the last week" or "number of transactions in the last hour" — that's where real fraud detection systems get most of their signal
