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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, accuracy_score, precision_score, recall_score, f1_score
)
import warnings
warnings.filterwarnings('ignore')

# Set styling
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
print("="*80)
print("TELECOM CUSTOMER CHURN ANALYSIS - COMPREHENSIVE ML PIPELINE")
print("="*80)

# ============================================================================
# 1. LOAD DATASET
# ============================================================================
print("\n1. LOADING DATA...")
try:
    df = pd.read_csv('TelecomCustomerChurn.csv')
    print(f"   ✓ Dataset loaded successfully")
    print(f"   Shape: {df.shape}")
    print(f"\n   First few rows:")
    print(df.head())
except FileNotFoundError:
    print("   ✗ ERROR: TelecomCustomerChurn.csv not found!")
    exit(1)

# ============================================================================
# 2. DATA EXPLORATION & OVERVIEW
# ============================================================================
print("\n2. DATA EXPLORATION & OVERVIEW...")
print(f"\n   Missing Values:")
missing_values = df.isnull().sum()
if missing_values.sum() > 0:
    print(missing_values[missing_values > 0])
else:
    print("   ✓ No missing values found")

print(f"\n   Data Types:")
print(df.dtypes)

print(f"\n   Churn Distribution:")
churn_dist = df['Churn'].value_counts()
print(churn_dist)
churn_rate = (df['Churn']=='Yes').sum()/len(df)*100
print(f"   Churn Rate: {churn_rate:.2f}%")

print(f"\n   Statistical Summary:")
print(df.describe())

# ============================================================================
# 3. DATA PREPROCESSING
# ============================================================================
print("\n3. DATA PREPROCESSING...")

# Store original data
df_original = df.copy()

# Encode target variable
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
print("   ✓ Target variable (Churn) encoded")

# Encode categorical variables
label_encoders = {}
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print(f"   ✓ {len(categorical_cols)} categorical variables encoded")

# ============================================================================
# 4. FEATURE ENGINEERING
# ============================================================================
print("\n4. FEATURE ENGINEERING...")
X = df.drop('Churn', axis=1)
y = df['Churn']

print(f"   Features shape: {X.shape}")
print(f"   Target shape: {y.shape}")
print(f"   Feature columns: {list(X.columns[:5])}... and more")

# ============================================================================
# 5. TRAIN-TEST SPLIT
# ============================================================================
print("\n5. SPLITTING DATA INTO TRAIN-TEST SETS...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Training set: {X_train.shape}")
print(f"   Testing set: {X_test.shape}")
print(f"   Class distribution in training set:")
for churn_val, count in y_train.value_counts().items():
    label = "Churn" if churn_val == 1 else "No Churn"
    percentage = (count / len(y_train)) * 100
    print(f"     {label}: {count} ({percentage:.1f}%)")

# ============================================================================
# 6. FEATURE SCALING
# ============================================================================
print("\n6. FEATURE SCALING...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("   ✓ Features scaled successfully using StandardScaler")

# ============================================================================
# 7. MODEL TRAINING
# ============================================================================
print("\n7. MODEL TRAINING...")

# Logistic Regression
print("\n   a) LOGISTIC REGRESSION")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
lr_score = lr_model.score(X_test_scaled, y_test)
print(f"      ✓ Model trained")
print(f"      Accuracy: {lr_score:.4f}")

# Random Forest Classifier
print("\n   b) RANDOM FOREST CLASSIFIER")
rf_model = RandomForestClassifier(
    n_estimators=100, 
    random_state=42, 
    n_jobs=-1,
    max_depth=20
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
rf_score = rf_model.score(X_test, y_test)
print(f"      ✓ Model trained")
print(f"      Accuracy: {rf_score:.4f}")

# ============================================================================
# 8. CROSS-VALIDATION
# ============================================================================
print("\n8. CROSS-VALIDATION ANALYSIS...")
lr_cv_scores = cross_val_score(LogisticRegression(max_iter=1000), X_train_scaled, y_train, cv=5)
rf_cv_scores = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42), X_train, y_train, cv=5)

print(f"   Logistic Regression CV Scores: {[f'{score:.4f}' for score in lr_cv_scores]}")
print(f"   Mean: {lr_cv_scores.mean():.4f} (+/- {lr_cv_scores.std():.4f})")

print(f"\n   Random Forest CV Scores: {[f'{score:.4f}' for score in rf_cv_scores]}")
print(f"   Mean: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std():.4f})")

# ============================================================================
# 9. MODEL EVALUATION
# ============================================================================
print("\n9. DETAILED MODEL EVALUATION...")

print("\n" + "="*80)
print("LOGISTIC REGRESSION - PERFORMANCE METRICS")
print("="*80)
print(f"\n   Classification Report:")
print(classification_report(y_test, lr_pred, target_names=['No Churn', 'Churn']))

lr_accuracy = accuracy_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred)
lr_recall = recall_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)

print(f"\n   Summary Metrics:")
print(f"   Accuracy:  {lr_accuracy:.4f}")
print(f"   Precision: {lr_precision:.4f}")
print(f"   Recall:    {lr_recall:.4f}")
print(f"   F1-Score:  {lr_f1:.4f}")

print("\n" + "="*80)
print("RANDOM FOREST - PERFORMANCE METRICS")
print("="*80)
print(f"\n   Classification Report:")
print(classification_report(y_test, rf_pred, target_names=['No Churn', 'Churn']))

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

print(f"\n   Summary Metrics:")
print(f"   Accuracy:  {rf_accuracy:.4f}")
print(f"   Precision: {rf_precision:.4f}")
print(f"   Recall:    {rf_recall:.4f}")
print(f"   F1-Score:  {rf_f1:.4f}")

# ============================================================================
# 10. ROC-AUC ANALYSIS
# ============================================================================
print("\n10. ROC-AUC ANALYSIS...")

lr_auc = roc_auc_score(y_test, lr_pred_proba)
rf_auc = roc_auc_score(y_test, rf_pred_proba)

print(f"\n   ROC-AUC Scores:")
print(f"   Logistic Regression AUC: {lr_auc:.4f}")
print(f"   Random Forest AUC:       {rf_auc:.4f}")

# ============================================================================
# 11. CONFUSION MATRIX ANALYSIS
# ============================================================================
print("\n11. CONFUSION MATRIX ANALYSIS...")

lr_cm = confusion_matrix(y_test, lr_pred)
rf_cm = confusion_matrix(y_test, rf_pred)

print(f"\n   Logistic Regression Confusion Matrix:")
print(lr_cm)
print(f"   (True Negatives, False Positives | False Negatives, True Positives)")

print(f"\n   Random Forest Confusion Matrix:")
print(rf_cm)
print(f"   (True Negatives, False Positives | False Negatives, True Positives)")

# ============================================================================
# 12. FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n12. FEATURE IMPORTANCE ANALYSIS...")

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\n   Top 15 Important Features (Random Forest):")
print(feature_importance.head(15))

# ============================================================================
# 13. MODEL COMPARISON & RECOMMENDATIONS
# ============================================================================
print("\n13. MODEL COMPARISON & RECOMMENDATIONS...")
print("\n" + "="*80)

if rf_auc > lr_auc:
    best_model = "Random Forest"
    best_auc = rf_auc
else:
    best_model = "Logistic Regression"
    best_auc = lr_auc

print(f"\n   RECOMMENDED MODEL: {best_model}")
print(f"   Best AUC Score: {best_auc:.4f}")
print(f"\n   Model Comparison:")
print(f"   {'Metric':<20} {'Logistic Regression':<25} {'Random Forest':<25}")
print(f"   {'-'*70}")
print(f"   {'Accuracy':<20} {lr_accuracy:<25.4f} {rf_accuracy:<25.4f}")
print(f"   {'Precision':<20} {lr_precision:<25.4f} {rf_precision:<25.4f}")
print(f"   {'Recall':<20} {lr_recall:<25.4f} {rf_recall:<25.4f}")
print(f"   {'F1-Score':<20} {lr_f1:<25.4f} {rf_f1:<25.4f}")
print(f"   {'AUC-ROC':<20} {lr_auc:<25.4f} {rf_auc:<25.4f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE - ALL MODELS SUCCESSFULLY TRAINED AND EVALUATED")
print("="*80)
