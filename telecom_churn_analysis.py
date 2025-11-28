"""
Telecom Customer Churn Analysis
Author: Bhupendra
Date: 2025
Description: Comprehensive ML analysis to predict customer churn in telecom industry
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Set styling
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("="*60)
print("TELECOM CUSTOMER CHURN ANALYSIS")
print("="*60)

# Load Dataset
print("\n1. LOADING DATA...")
df = pd.read_csv('TelecomCustomerChurn.csv')
print(f"Dataset Shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

# Data Overview
print("\n2. DATA OVERVIEW...")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nChurn Distribution:")
print(df['Churn'].value_counts())
print(f"\nChurn Rate: {(df['Churn']=='Yes').sum()/len(df)*100:.2f}%")

# Exploratory Data Analysis
print("\n3. EXPLORATORY DATA ANALYSIS...")
print(f"\nStatistical Summary:")
print(df.describe())

# Data Preprocessing
print("\n4. DATA PREPROCESSING...")
# Encode target variable
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Encode categorical variables
label_encoders = {}
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print("Categorical variables encoded.")

# Feature Engineering
print("\n5. FEATURE ENGINEERING...")
X = df.drop('Churn', axis=1)
y = df['Churn']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split Data
print("\n6. SPLITTING DATA...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set: {X_train.shape}")
print(f"Testing set: {X_test.shape}")

# Scale Features
print("\n7. FEATURE SCALING...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled successfully.")

# Model Training
print("\n8. MODEL TRAINING...")

# Logistic Regression
print("\n   a) Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_score = lr_model.score(X_test_scaled, y_test)
print(f"   Accuracy: {lr_score:.4f}")

# Random Forest
print("\n   b) Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_score = rf_model.score(X_test, y_test)
print(f"   Accuracy: {rf_score:.4f}")

# Model Evaluation
print("\n9. MODEL EVALUATION...")
print("\n   Logistic Regression Report:")
print(classification_report(y_test, lr_pred, target_names=['No Churn', 'Churn']))

print("\n   Random Forest Report:")
print(classification_report(y_test, rf_pred, target_names=['No Churn', 'Churn']))

# ROC-AUC Score
lr_auc = roc_auc_score(y_test, lr_model.predict_proba(X_test_scaled)[:, 1])
rf_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])

print(f"\n   ROC-AUC Scores:")
print(f"   Logistic Regression AUC: {lr_auc:.4f}")
print(f"   Random Forest AUC: {rf_auc:.4f}")

# Feature Importance
print("\n10. FEATURE IMPORTANCE ANALYSIS...")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
