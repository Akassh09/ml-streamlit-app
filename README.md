# ML Classification Models - Assignment 2

## a. Problem statement
The objective of this assignment is to implement and compare multiple classification models to predict whether an individual earns more than $50K per year using a public classification dataset.

The app is built in Streamlit and allows users to upload a CSV test file, select a model, and view evaluation metrics, confusion matrix, and classification report.

## b. Dataset description
The chosen dataset is the Adult (Census Income) dataset, a binary classification dataset downloaded from OpenML/UCI.

Dataset details:
- Type: Binary classification
- Number of instances: More than 500 (Adult dataset has 48,842 rows)
- Number of features: More than 12 features
- Target variable: `income`
- Target labels: `<=50K` and `>50K`

The dataset contains demographic and employment-related features such as age, education, marital status, occupation, hours per week, capital gain/loss, and native country.

## c. GitHub Repository Link
GitHub Repository Link: https://github.com/Akassh09/ml-streamlit-app.git


## d. Models used
The following six classification models were implemented and evaluated on the same dataset:
- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbor Classifier
- Naive Bayes Classifier
- Random Forest Classifier (Ensemble)

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.84599 | 0.90465 | 0.73159 | 0.59813 | 0.65816 | 0.56479 |
| Decision Tree | 0.80597 | 0.74733 | 0.60393 | 0.63113 | 0.61723 | 0.48757 |
| kNN | 0.82488 | 0.85203 | 0.66566 | 0.58965 | 0.62535 | 0.51323 |
| Naive Bayes | 0.64478 | 0.85068 | 0.40464 | 0.91882 | 0.56184 | 0.41197 |
| Random Forest (Ensemble) | 0.84456 | 0.89978 | 0.72093 | 0.60839 | 0.65989 | 0.56347 |

### Observations on model performance

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Performed best overall with strong balance between accuracy, AUC, precision, and MCC. Good generalization for the chosen dataset. |
| Decision Tree | Simple and interpretable, but overfitting risk is visible and performance is weaker than logistic regression and random forest. |
| kNN | Reasonably good accuracy, but it is sensitive to feature scaling and showed slightly lower precision/recall balance compared to logistic regression. |
| Naive Bayes | Very high recall but poor precision and lower accuracy, which indicates it tends to classify more samples as positive. |
| Random Forest (Ensemble) | Strong performance, close to logistic regression, and robust due to ensemble averaging; good precision and F1 score. |

### Overall Winner for the dataset
Overall Winner: Logistic Regression

Reason: It achieved the highest AUC and the best overall trade-off among Accuracy, Precision, F1, and MCC on this dataset.

## How to run the project
1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train the machine learning models:
   ```bash
   python train_models.py
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Project files
- `app.py` — Streamlit dashboard
- `train_models.py` — model training and dataset handling
- `requirements.txt` — required Python packages
- `model/` — saved trained models
- `test_data.csv` — sample test dataset
- `README.md` — project documentation

