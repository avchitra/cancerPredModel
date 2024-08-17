# Cancer Risk Analysis and Prediction Model

## Overview
This project analyzes the risk of developing or dying from various types of cancer based on statistical data. It uses a logistic regression model to classify cancer risks as high or low.

### Data Description
The dataset "Risk of Developing or Dying From Cancer.csv" contains information about different cancer types and their associated risks for both males and females. The data includes:

Cancer Type
Risk of developing cancer (percentage and ratio) for males and females
Risk of dying from cancer (percentage and ratio) for males and females

## Code Analysis

### 1. Data Preparation

The code reads the CSV file into a pandas DataFrame.
New features are created for development and death risks for both genders.
The target variable 'High_Risk' is created, defined as 1 if the death risk is > 1% for either gender, 0 otherwise.
Data is normalized using StandardScaler.
The dataset is split into training (80%) and testing (20%) sets, with stratification to ensure balanced classes.

### 2. Feature Selection

Recursive Feature Elimination (RFE) is used to select the most important features.
The selected features are printed for transparency.

### 3. Model Training

A logistic regression model is trained on the selected features.

### 4. Model Evaluation

5-fold cross-validation is performed to assess model performance.
The model is evaluated on the test set using classification report and confusion matrix.

### 5. Model Interpretation

SHAP (SHapley Additive exPlanations) values are used to interpret the model's predictions.
Feature importance is calculated and printed.

### 6. Prediction on New Data

The model can make predictions on new, unseen data.

Improvements Implemented

Data Balance: The target variable is now based on a threshold (1% death risk), creating a more balanced dataset.
Feature Engineering: New features were created, and feature selection was implemented using RFE.
Model Selection: A simpler, more interpretable logistic regression model is used instead of a neural network.
Cross-Validation: 5-fold cross-validation is implemented for more robust performance evaluation.
Interpretability: SHAP values are used to interpret the model's predictions and feature importance.

## Results and Interpretation
- Accuracy: 80%
- Weighted Precision: 80%
- Weighted Recall: 64%
- Weighted F1: 71%


The model's performance metrics (accuracy, precision, recall, F1-score) are reported.
The most important features for predicting high cancer risk are identified.
The model's ability to generalize is assessed through cross-validation scores.

## Conclusion
This improved implementation provides a more robust and interpretable analysis of cancer risk factors. The logistic regression model, combined with feature selection and SHAP values, offers insights into which factors are most predictive of high cancer risk. However, it's important to note that this model is based on population-level statistics and should not be used for individual medical diagnosis or treatment decisions.

### Potential Improvements
- Expand on the data
    - More demographic data
    - Genetic information
    - True negative cases
- Model Comparison
    - Explore Other Models(Random Forests, Gradient Boosting, SVMs) might yield better performance or results
    - Ensemble Methods: Combining multiply models could improve performance
- Feature Engineering
    - Polynomial Features: Could capture non-linear relationships in the data
- Fine-Tuning and Hyperparameter Optimization
    - Grid / Random Search: Utilize these methods to optimize hyperparams for the regression model or others
    - Regularization: Experiment with different regularization techniques (L1, L2) to improve model generalization
- Model Validation
    - Test the model on an external dataset to ensure generalizability


## Future Work
Incorporate more detailed demographic data if available.
Explore other machine learning models and compare their performance.
Conduct a more detailed analysis of specific cancer types.

## Authors

- RishikPamuru
- avchitra