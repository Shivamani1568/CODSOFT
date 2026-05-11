# Task 2: Customer Churn Prediction

For this task I had to build a model that predicts whether a customer will leave a subscription service. I used the bank churn dataset which has 10,000 customers along with their demographics, account info, and a flag for whether they ended up leaving.

## The data

Each row is one customer. The features are credit score, age, tenure (years with the bank), account balance, number of products held, credit card status, whether they're an active member, estimated salary, country (France, Germany, or Spain), and gender. The target column is Exited, where 1 means the customer left and 0 means they stayed. About 20% of customers churned so the dataset is imbalanced.

The CSV isn't pushed to the repo because data files are in .gitignore. It's the standard Churn_Modelling.csv dataset, easy to find on Kaggle.

## What I did

First I dropped RowNumber, CustomerId, and Surname since they don't help with prediction. Then I built a preprocessing pipeline that scales the numeric columns with StandardScaler and one-hot encodes Geography and Gender. I split the data 80/20 with stratification so both train and test sets have the same churn ratio.

Then I trained three models and compared them:

- Logistic Regression with class_weight balanced (to handle the imbalance)
- Random Forest with 200 trees and max depth 10
- Gradient Boosting with 200 estimators and learning rate 0.1

For each model I measured accuracy, precision, recall, F1 score, and ROC-AUC on the test set.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.714 | 0.388 | 0.700 | 0.499 | 0.777 |
| Random Forest | 0.839 | 0.596 | 0.646 | 0.620 | 0.861 |
| Gradient Boosting | 0.864 | 0.759 | 0.487 | 0.593 | 0.865 |

Gradient Boosting came out on top by ROC-AUC. I picked AUC as the main metric because the classes are imbalanced. A model that just predicts stayed for everyone would get around 80% accuracy and be useless, so accuracy alone is misleading here.

The thing that surprised me was how differently the three models behave even though their AUC scores are close. Logistic Regression catches 70% of actual churners but has bad precision (only 39%), meaning it raises a lot of false alarms. Gradient Boosting is the opposite, when it predicts churn it's right 76% of the time but only catches about half of the actual churners. So picking the best model really depends on what you do with the prediction. If you're sending discount emails, recall matters more. If you're calling each flagged customer personally, precision matters more because each call has a cost.

## Top features

From the Gradient Boosting feature importances:

1. Age, older customers churn more
2. NumOfProducts, customers with 3+ products churn a lot, which I didn't expect at first
3. Balance, higher balances tend to leave more
4. IsActiveMember, inactive members are easy to lose
5. Geography_Germany, German customers churn way more than French or Spanish ones

The full plot is saved as churn_results.png. It has the model comparison, ROC curves, confusion matrix, and feature importance all in one figure.

## How to run

From inside this folder:

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    python churn_prediction.py

You need Churn_Modelling.csv in the same folder. The script prints results to the terminal and saves churn_results.png.

## What I would try next

If I had more time I would try XGBoost or LightGBM since they usually beat sklearn's GradientBoosting on tabular data by a couple of AUC points. I would also try SMOTE to handle the class imbalance more aggressively, and use GridSearchCV for proper hyperparameter tuning instead of the rough defaults I picked.
