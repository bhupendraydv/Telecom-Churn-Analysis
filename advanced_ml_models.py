"""Advanced ML Models for Telecom Churn - XGBoost & Ensemble"""
import pandas as pd, numpy as np, joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

class AdvancedChurnModels:
    def __init__(self, df, target='Churn'):
        self.df = df
        self.target = target
        self.models = {}
        self.scalers, self.encoders = {}, {}
    
    def preprocess(self):
        df = self.df.copy()
        df = df.fillna(df.mean(numeric_only=True))
        for col in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.encoders[col] = le
        X, y = df.drop(self.target, axis=1), df[self.target]
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        self.scalers['features'] = scaler
        return train_test_split(X, y, test_size=0.2, random_state=42), X, y
    
    def train_xgboost(self, X_train, y_train):
        model = xgb.XGBClassifier(n_estimators=200, max_depth=7, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train, verbose=False)
        self.models['xgboost'] = model
        return model
    
    def train_lightgbm(self, X_train, y_train):
        model = lgb.LGBMClassifier(n_estimators=200, max_depth=7, random_state=42)
        model.fit(X_train, y_train, verbose=-1)
        self.models['lightgbm'] = model
        return model
    
    def train_ensemble(self, X_train, y_train):
        ensemble = VotingClassifier([
            ('xgb', self.models['xgboost']),
            ('lgb', self.models['lightgbm']),
            ('gb', GradientBoostingClassifier(n_estimators=100)),
            ('rf', RandomForestClassifier(n_estimators=100))
        ], voting='soft')
        ensemble.fit(X_train, y_train)
        self.models['ensemble'] = ensemble
        return ensemble
    
    def evaluate(self, X_test, y_test):
        results = {}
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            results[name] = {'acc': (y_pred == y_test).mean(), 'auc': roc_auc_score(y_test, y_proba)}
        return results
    
    def save_models(self, path='models/'):
        import os
        os.makedirs(path, exist_ok=True)
        for name, model in self.models.items():
            joblib.dump(model, f'{path}{name}.pkl')
        joblib.dump(self.scalers, f'{path}scalers.pkl')
        print(f'✓ Models saved to {path}')
