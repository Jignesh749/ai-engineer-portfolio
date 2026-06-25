import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier

st.set_page_config(page_title="Credit Risk Model", layout="centered")
st.title("Credit Risk Model")
st.write("Predict whether a borrower is likely to default using the German Credit dataset.")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "german_credit_data.csv"

@st.cache_data
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df["target"] = df["Risk"].map({"good": 0, "bad": 1})
    df = df.drop(columns=["Risk"])
    return df

def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

def train_models(df):
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(X_train)

    logistic = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

    xgb = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        ))
    ])

    models = {"Logistic Regression": logistic, "XGBoost": xgb}
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "auc": roc_auc_score(y_test, probs),
            "report": classification_report(y_test, preds, output_dict=True),
            "cm": confusion_matrix(y_test, preds)
        }

    return results

df = load_data()
results = train_models(df)

c1, c2 = st.columns(2)
with c1:
    st.metric("Logistic AUC", f"{results['Logistic Regression']['auc']:.4f}")
with c2:
    st.metric("XGBoost AUC", f"{results['XGBoost']['auc']:.4f}")

st.subheader("Model Comparison")
summary = pd.DataFrame({
    "model": ["Logistic Regression", "XGBoost"],
    "roc_auc": [results["Logistic Regression"]["auc"], results["XGBoost"]["auc"]],
    "accuracy": [results["Logistic Regression"]["report"]["accuracy"], results["XGBoost"]["report"]["accuracy"]],
    "default_recall": [results["Logistic Regression"]["report"]["1"]["recall"], results["XGBoost"]["report"]["1"]["recall"]],
    "default_precision": [results["Logistic Regression"]["report"]["1"]["precision"], results["XGBoost"]["report"]["1"]["precision"]],
})
st.dataframe(summary, use_container_width=True)

st.subheader("Confusion Matrices")
st.write("Logistic Regression")
st.write(results["Logistic Regression"]["cm"])
st.write("XGBoost")
st.write(results["XGBoost"]["cm"])

st.subheader("Dataset Preview")
st.dataframe(df.head(), use_container_width=True)