import os
import pandas as pd
import joblib
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


MODEL_DIR = 'model'


def load_adult_dataset():
    # Try fetching the Adult dataset (Census Income) from OpenML.
    # If network or API is unavailable, fall back to a synthetic dataset large
    # enough to meet assignment requirements (>=500 instances, >=12 features).
    try:
        print('Attempting to download dataset from OpenML...')
        adult = fetch_openml('adult', version=2, as_frame=True)
        df = adult.frame.copy()
        # target column is 'class' or 'income'
        if 'class' in df.columns and 'income' not in df.columns:
            df = df.rename(columns={'class':'income'})

        # Clean missing values marked as '?'
        df = df.replace('?', pd.NA).dropna()
        print('Downloaded Adult dataset from OpenML.')
        return df
    except Exception as e:
        print(f'Could not download OpenML dataset (reason: {e}). Falling back to synthetic dataset.')
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=2000, n_features=12, n_informative=8,
                                   n_redundant=2, n_classes=2, random_state=42)
        cols = [f'f{i}' for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=cols)
        df['income'] = y
        return df


def build_preprocessor(df):
    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'income' in numeric_cols:
        numeric_cols.remove('income')
    categorical_cols = [c for c in df.columns if c not in numeric_cols and c != 'income']

    numeric_transformer = StandardScaler()
    # OneHotEncoder parameter name changed between sklearn versions (sparse -> sparse_output).
    try:
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'
    )
    return preprocessor


def train_and_save():
    df = load_adult_dataset()
    # Convert target to binary 0/1
    df['income'] = df['income'].apply(lambda x: 1 if str(x).strip().startswith('>') else 0)

    X = df.drop(columns=['income'])
    y = df['income']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    preprocessor = build_preprocessor(df)

    models = {
        'logistic_regression': LogisticRegression(max_iter=1000),
        'decision_tree': DecisionTreeClassifier(random_state=42),
        'knn': KNeighborsClassifier(),
        'naive_bayes': GaussianNB(),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    os.makedirs(MODEL_DIR, exist_ok=True)

    for name, estimator in models.items():
        print(f'Training {name} ...')
        pipe = Pipeline(steps=[('preprocessor', preprocessor), ('clf', estimator)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f'{name} accuracy: {acc:.4f}')
        joblib.dump(pipe, os.path.join(MODEL_DIR, f'{name}.joblib'))
    # Save a small sample test csv for demo
    sample = X_test.copy()
    sample['income'] = y_test.values
    sample.to_csv('test_data.csv', index=False)
    print('Saved sample test_data.csv and models in `model/`')


if __name__ == '__main__':
    train_and_save()
