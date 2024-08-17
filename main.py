import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import RFE
import shap #type: ignore


# 1. Data Preparation
df = pd.read_csv("cancerdata.csv")

# Create features for both genders
for gender in ['Male', 'Female']:
    df[f'{gender}_Dev_Risk'] = df[f'{gender} Risk Development Percentage'] / 100
    df[f'{gender}_Death_Risk'] = df[f'{gender} Risk Dying Percentage'] / 100

# Create target variable (1 if death risk > 1%, 0 otherwise)
df['High_Risk'] = ((df['Male_Death_Risk'] > 0.01) | (df['Female_Death_Risk'] > 0.01)).astype(int)

# Select relevant features
features = ['Male_Dev_Risk', 'Male_Death_Risk', 'Female_Dev_Risk', 'Female_Death_Risk']
X = df[features]
y = df['High_Risk']

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 2. Feature Selection
logreg = LogisticRegression(random_state=42)
rfe = RFE(estimator=logreg, n_features_to_select=3)
X_train_rfe = rfe.fit_transform(X_train, y_train)
X_test_rfe = rfe.transform(X_test)

# Get selected feature names
selected_features = [features[i] for i in range(len(features)) if rfe.support_[i]]
print("Selected features:", selected_features)

# 3. Model Training
model = LogisticRegression(random_state=42)
model.fit(X_train_rfe, y_train)

# 4. Model Evaluation
# Cross-validation
cv_scores = cross_val_score(model, X_train_rfe, y_train, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Evaluation on test set
y_pred = model.predict(X_test_rfe)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 5. Model Interpretation
explainer = shap.LinearExplainer(model, X_train_rfe)
shap_values = explainer.shap_values(X_test_rfe)

print("\nFeature Importance:")
for i, feature in enumerate(selected_features):
    print(f"{feature}: {abs(shap_values[:, i].mean()):.4f}")

# 6. Predict on new data
new_data = np.array([[0.05, 0.02, 0.04, 0.015]])  # Example new data point
new_data_scaled = scaler.transform(new_data)
new_data_rfe = rfe.transform(new_data_scaled)
prediction = model.predict(new_data_rfe)
print(f"\nPrediction for new data: {'High Risk' if prediction[0] == 1 else 'Low Risk'}")