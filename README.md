# Telco-churn-prediction
ML pipeline to predict customer churn on the Telco dataset — includes SMOTE, hyperparameter tuning, and SHAP explainability (XGBoost, ROC-AUC 0.83).

It is a machine learning project that predicts telecom customer churn using the IBM/Kaggle **Telco Customer Churn** dataset. The notebook walks through data cleaning, feature engineering, model building, hyperparameter tuning, interpretability (SHAP), and business-focused risk segmentation.

## Overview

Customer churn — when a customer stops doing business with a company — is costly to acquire back. This project builds and compares classification models to predict which customers are likely to churn, so a business can proactively target retention efforts at high-risk customers.

**Dataset:** `Telco_customer_churn.csv` — 7,043 customers, 33 original columns
**Target variable:** `Churn Value` (1 = churned, 0 = retained)

## Features Used

**Numeric:**
- Monthly Charges
- Total Charges
- Tenure Months

**Categorical:**
- Gender, Senior Citizen, Partner, Dependents
- Phone Service, Multiple Lines, Internet Service
- Online Security, Online Backup, Device Protection, Tech Support
- Streaming TV, Streaming Movies
- Contract, Paperless Billing, Payment Method

## Workflow

1. **Data Cleaning** — Converted `Total Charges` to numeric and dropped rows with invalid/missing values.
2. **Train/Test Split** — 70/30 stratified split to preserve the churn class ratio.
3. **Feature Selection** — Used `SelectKBest` with mutual information to identify the most predictive features.
4. **Preprocessing Pipeline** — Built with `ColumnTransformer`:
   - Numeric: median imputation + standard scaling
   - Categorical: most-frequent imputation + one-hot encoding
5. **Class Imbalance Handling** — Applied **SMOTE** oversampling and `class_weight='balanced'` to address the skewed churn/no-churn ratio.
6. **Model Training** — Logistic Regression, Random Forest, and XGBoost, each wrapped in a full `sklearn` pipeline.
7. **Hyperparameter Tuning** — `GridSearchCV` (ROC-AUC scoring, cross-validation) for both Random Forest and XGBoost.
8. **Evaluation** — Accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices, and ROC curve comparisons across models.
9. **Interpretability** — Feature importance ranking from Random Forest, plus **SHAP** summary plots to explain individual predictions.
10. **Threshold Analysis** — Swept classification thresholds to study the precision/recall trade-off instead of relying on the default 0.5 cutoff.
11. **Risk Segmentation** — Bucketed customers into Low / Medium / High churn-risk tiers based on predicted probability, for business use.

## Model Performance

| Model                    | ROC AUC | F1 Score | Precision | Recall |
|---------------------------|---------|----------|-----------|--------|
| Logistic Regression       | 0.71    | 0.50     | 0.41      | 0.63   |
| Random Forest (Tuned)     | 0.75    | 0.00     | 0.00      | 0.00   |
| XGBoost (Initial)         | 0.76    | 0.00     | 0.00      | 0.00   |
| **XGBoost (Tuned)**       | **0.83**| **0.61** | **0.52**  | **0.74** |

The tuned XGBoost model performed best overall. Note that Random Forest (Tuned) and XGBoost (Initial) show a 0 F1/precision/recall at the default 0.5 threshold — an artifact of the class imbalance, which motivated the threshold-tuning step in the notebook.

## Key Insights

- Contract type, tenure, and monthly charges are among the strongest drivers of churn.
- Month-to-month contract customers churn far more than those on longer-term contracts.
- SHAP analysis confirms that tenure and contract type consistently push predictions toward "churn" for short-tenure, month-to-month customers.

## Tech Stack

- **Language:** Python
- **Data handling:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **ML:** scikit-learn, XGBoost, imbalanced-learn (SMOTE)
- **Interpretability:** SHAP

## Project Structure

```
├── Jhanvi_Project_Customer_Churn_Prediction.ipynb   # Full analysis and modeling notebook
├── Telco_customer_churn.csv                          # Dataset (not included — see below)
└── README.md
```

## Getting Started

1. Clone the repository and install dependencies:
   ```bash
   pip install pandas numpy seaborn matplotlib scikit-learn xgboost imbalanced-learn shap
   ```
2. Download the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it in the project directory.
3. Open and run `Jhanvi_Project_Customer_Churn_Prediction.ipynb` in Jupyter or Google Colab.

## Future Improvements

- Deploy the tuned XGBoost model behind a simple API for real-time churn scoring.
- Experiment with additional models (LightGBM, CatBoost) and ensembling.
- Incorporate cost-sensitive evaluation tied to actual retention-offer economics.

## Author

**Jhanvi Khanna**  
