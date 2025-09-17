# Airline Passenger Satisfaction Prediction

## 📖 Overview

This project builds a machine learning model to predict passenger satisfaction with airline services based on survey data, including factors like flight delays, seat comfort, and in-flight entertainment. The goal is to classify satisfaction levels (satisfied/neutral/dissatisfied) to help airlines improve customer experience. The implementation uses **pandas** for data processing, **scikit-learn** for preprocessing and modeling, and **XGBoost** or similar for classification, with hyperparameter tuning via cross-validation.

The project is implemented in a Jupyter Notebook (`satisfaction.ipynb`) and generates predictions for a test dataset, saved in `submission.csv` for evaluation.

## 🎯 Objectives

- **Predict Satisfaction**: Classify passenger satisfaction as binary or multi-class based on survey responses.
- **Preprocess Survey Data**: Handle categorical (e.g., class, route) and numerical features (e.g., delays, ratings).
- **Model Optimization**: Use ensemble methods to achieve high accuracy (target: &gt;85% on validation).
- **Generate Submission**: Output predicted labels/probabilities for the test set.

## ✨ Features

- **Data Preprocessing**:
  - One-hot encoding for categorical variables (e.g., `CustomerType`, `TypeOfTravel`).
  - Scaling numerical features (e.g., `DepartureDelay`, `ArrivalDelay`).
  - Handling missing values and outliers in delay times.
- **Model Training**:
  - **XGBoost Classifier** for gradient boosting on imbalanced data.
  - Cross-validation with ROC-AUC or accuracy scoring.
  - Hyperparameter tuning (e.g., learning rate, max depth).
- **Evaluation**:
  - Confusion matrix, classification report, ROC curve.
  - Feature importance analysis to identify key satisfaction drivers (e.g., delays, entertainment).
- **Submission**:
  - Predicted satisfaction labels for test data.
  - Zipped output including notebook and CSV.

## 🛠 Prerequisites

- **Input Files** (in `data/`):
  - `train.csv`: Training data with satisfaction labels.
  - `test.csv`: Test data for predictions.

## 🚀 Usage

1. Open `satisfaction.ipynb` in Jupyter Notebook.
2. Run cells sequentially:
   - Load and explore data (EDA: distributions, correlations).
   - Preprocess: Encode categoricals, scale numerics, split train/validation.
   - Train model with CV and tune hyperparameters.
   - Evaluate on validation set (metrics, plots).
   - Predict on test set and generate submission.
3. Output:
   - `submission.csv`: Predicted satisfaction labels.
   - `result.zip`: Zipped notebook and CSV for submission.

### Key Code Snippets

- **Data Loading & EDA**:

  ```python
  import pandas as pd
  train = pd.read_csv('data/train.csv')
  print(train['satisfaction'].value_counts())  # Check class balance
  ```
- **Preprocessing**:

  ```python
  from sklearn.preprocessing import OneHotEncoder, StandardScaler
  encoder = OneHotEncoder(sparse=False)
  scaler = StandardScaler()
  # Fit and transform...
  ```
- **Model Training**:

  ```python
  from xgboost import XGBClassifier
  from sklearn.model_selection import cross_val_score
  model = XGBClassifier(random_state=42)
  scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
  ```
- **Submission**:

  ```python
  preds = model.predict(X_test)
  submission = pd.DataFrame({'id': test_id, 'satisfaction': preds})
  submission.to_csv('submission.csv', index=False)
  ```

## 📊 Code Structure

- **Cells 1-4**: Import libraries, load data, EDA (plots, stats).
- **Cells 5-7**: Preprocessing (encoding, scaling, handling imbalances).
- **Cells 8-10**: Train-validation split, model training with CV.
- **Cells 11-13**: Evaluation (metrics, feature importance, plots).
- **Cell 14**: Test predictions, submission generation, zip.

## 🔍 Evaluation

- **Metrics**: Accuracy, F1-score (weighted for imbalance), ROC-AUC.
- **Cross-Validation**: 5-fold CV for robust performance estimation.
- **Thresholding**: Optimal cutoff from ROC curve for binary decisions.
- **Insights**: Delays and entertainment often top feature importances.

## 📝 Notes

- **Imbalance Handling**: Use class weights in XGBoost for minority classes (e.g., dissatisfied).
- **Feature Engineering**: Derived features like total delay or service score.
- **Scalability**: Efficient for large datasets; GPU acceleration via XGBoost.
- **Improvements**: Ensemble with RandomForest, SHAP for explainability.