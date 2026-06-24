# Credit Risk Model

End-to-end credit risk classification project using the German Credit dataset.

## Goal
Predict whether a loan applicant is likely to default based on structured financial and demographic features.

## Dataset
- German Credit Data from Kaggle
- Target column: `Risk`
- Labels mapped as:
  - `good` = 0
  - `bad` = 1

## Workflow
1. Load the CSV dataset.
2. Drop unnecessary index columns.
3. Handle missing values.
4. Encode categorical features.
5. Train baseline Logistic Regression.
6. Train XGBoost classifier.
7. Evaluate using:
   - ROC AUC
   - Precision
   - Recall
   - F1-score
   - Confusion Matrix

## Results
- Logistic Regression ROC AUC: 0.6368
- XGBoost ROC AUC: 0.6913

## Notes
In credit risk, recall on the default class is important because missing a risky borrower is costly.