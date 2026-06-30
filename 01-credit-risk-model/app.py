import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier


def load_data(path='german_credit_data.csv'):
    df = pd.read_csv(path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    df['target'] = df['Risk'].map({'good': 0, 'bad': 1})
    df = df.drop(columns=['Risk'])
    return df

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )


def make_logistic_model(preprocessor):
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(max_iter=2000, class_weight='balanced'))
    ])


def make_xgb_model(preprocessor):
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        ))
    ])


def train_and_evaluate(path='german_credit_data.csv'):
    df = load_data(path)
    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(X_train)

    models = {
        'logistic': make_logistic_model(preprocessor),
        'xgboost': make_xgb_model(preprocessor)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        results[name] = {
            'report': classification_report(y_test, preds, output_dict=True),
            'auc': roc_auc_score(y_test, probs),
            'confusion_matrix': confusion_matrix(y_test, preds).tolist()
        }
        print(f'===== {name.upper()} =====')
        print('ROC AUC:', results[name]['auc'])
        print(classification_report(y_test, preds))
        print('Confusion Matrix:', results[name]['confusion_matrix'])
        print()

    return results


if __name__ == '__main__':
    train_and_evaluate()
