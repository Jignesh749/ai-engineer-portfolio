import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier


def preprocess_data(df, target_col='default'):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


def make_logistic_model():
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])


def make_xgb_model():
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )


def train_models(df, target_col='default'):
    X_train, X_test, y_train, y_test = preprocess_data(df, target_col)

    log_model = make_logistic_model()
    log_model.fit(X_train, y_train)
    log_pred = log_model.predict(X_test)
    log_prob = log_model.predict_proba(X_test)[:, 1]

    xgb_model = make_xgb_model()
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

    results = {
        'logistic': {
            'report': classification_report(y_test, log_pred, output_dict=True),
            'auc': roc_auc_score(y_test, log_prob),
            'confusion_matrix': confusion_matrix(y_test, log_pred).tolist(),
        },
        'xgboost': {
            'report': classification_report(y_test, xgb_pred, output_dict=True),
            'auc': roc_auc_score(y_test, xgb_prob),
            'confusion_matrix': confusion_matrix(y_test, xgb_pred).tolist(),
        }
    }
    return results, log_model, xgb_model


if __name__ == '__main__':
    print('Credit Risk Model ready for training.')