import streamlit as st
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)


MODEL_DIR = "model"
DEFAULT_TEST_CSV = "test_data.csv"


def load_models():
    models = {}
    model_path = os.path.join(os.path.dirname(__file__), MODEL_DIR)
    if not os.path.exists(model_path):
        return models
    for fname in os.listdir(model_path):
        if fname.endswith('.joblib'):
            path = os.path.join(model_path, fname)
            try:
                models[fname.replace('.joblib','')] = joblib.load(path)
            except Exception:
                st.warning(f"Could not load model: {fname}")
    return models


def compute_metrics(y_true, y_pred, y_proba=None):
    metrics = {}
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    if y_proba is not None:
        try:
            metrics['AUC'] = roc_auc_score(y_true, y_proba[:,1])
        except Exception:
            metrics['AUC'] = None
    else:
        metrics['AUC'] = None
    metrics['Precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['Recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['F1'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['MCC'] = matthews_corrcoef(y_true, y_pred)
    return metrics


def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    return fig


def main():
    st.title('ML Models Demo - Assignment 2')
    st.write('Upload a test CSV (or use provided sample). Select a trained model to evaluate.')

    # Ensure working dir is the script directory
    os.chdir(os.path.dirname(__file__))

    models = load_models()

    uploaded_file = st.file_uploader('Upload test CSV (only features + target column named `income` expected)')
    if uploaded_file is not None:
        try:
            test_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error('Failed to read uploaded CSV: ' + str(e))
            return
    else:
        if os.path.exists(DEFAULT_TEST_CSV):
            test_df = pd.read_csv(DEFAULT_TEST_CSV)
            st.info(f'Using sample test data from `{DEFAULT_TEST_CSV}`')
        else:
            st.warning('No test data found. Please upload a CSV or run training to generate models.')
            test_df = None

    st.sidebar.header('Model & Actions')
    model_names = list(models.keys())
    selected_model = st.sidebar.selectbox('Choose model', ['--'] + model_names)

    if st.sidebar.button('Train models (run once)'):
        st.info('Training models. This may take a few minutes.')
        import subprocess, sys
        subprocess.run([sys.executable, 'train_models.py'])
        st.experimental_rerun()

    if selected_model and selected_model != '--' and test_df is not None:
        model = models[selected_model]

        if 'income' not in test_df.columns:
            st.error('The test CSV must include a column named `income` as the target.')
            return

        X_test = test_df.drop(columns=['income'])
        y_test = test_df['income'].copy()
        # Normalize target to 0/1 if needed
        y_test = y_test.apply(lambda x: 1 if str(x).strip().startswith('>') or str(x).strip() in ['1','yes','>50K','>50k','>50K.'] else 0)

        try:
            y_pred = model.predict(X_test)
        except Exception as e:
            st.error('Model prediction failed: ' + str(e))
            return

        y_proba = None
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
        except Exception:
            y_proba = None

        metrics = compute_metrics(y_test, y_pred, y_proba)
        st.subheader('Evaluation Metrics')
        for k,v in metrics.items():
            st.write(f'- **{k}**: {v}')

        st.subheader('Confusion Matrix')
        fig = plot_confusion(y_test, y_pred)
        st.pyplot(fig)

        st.subheader('Classification Report')
        st.text(classification_report(y_test, y_pred, zero_division=0))

        if st.button('Show raw predictions'):
            out = X_test.copy()
            out['y_true'] = y_test
            out['y_pred'] = y_pred
            if y_proba is not None:
                out['proba_0'] = y_proba[:,0]
                out['proba_1'] = y_proba[:,1]
            st.dataframe(out)


if __name__ == '__main__':
    main()
