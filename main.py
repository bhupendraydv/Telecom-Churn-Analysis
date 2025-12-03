"""Main orchestrator for Telecom Churn Analysis system.

This module serves as the central hub for executing the complete ML pipeline:
- Loading and preprocessing data
- Training advanced ML models (XGBoost, LightGBM, Ensemble)
- Analyzing customer churn risk
- Generating automated reports
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from advanced_ml_models import AdvancedMLModels
from customer_risk_analyzer import CustomerRiskAnalyzer
from report_generator import ReportGenerator


class TelecomChurnPipeline:
    """Main ML pipeline orchestrator."""
    
    def __init__(self, data_path='TelecomCustomerChurn.csv'):
        """Initialize pipeline with data path."""
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.ml_models = None
        self.risk_analyzer = None
        self.report_gen = None
    
    def load_and_preprocess(self):
        """Load and preprocess data."""
        print("Loading data...")
        self.data = pd.read_csv(self.data_path)
        print(f"Data shape: {self.data.shape}")
        return self.data
    
    def train_models(self, test_size=0.2):
        """Train advanced ML models."""
        print("Preparing data for training...")
        X = self.data.drop('Churn', axis=1)
        y = self.data['Churn']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print("Training models...")
        self.ml_models = AdvancedMLModels()
        self.ml_models.train(X_train, y_train, X_test, y_test)
        
        return self.ml_models
    
    def analyze_risk(self):
        """Analyze customer churn risk."""
        print("Analyzing customer risk...")
        self.risk_analyzer = CustomerRiskAnalyzer(self.ml_models)
        risk_df = self.risk_analyzer.batch_score(self.data)
        return risk_df
    
    def generate_report(self, risk_df):
        """Generate automated report."""
        print("Generating report...")
        self.report_gen = ReportGenerator()
        report = self.report_gen.generate_full_report(
            self.data, risk_df, self.ml_models
        )
        return report
    
    def run_pipeline(self):
        """Execute complete pipeline."""
        print("Starting Telecom Churn Analysis Pipeline...\n")
        
        self.load_and_preprocess()
        self.train_models()
        risk_df = self.analyze_risk()
        report = self.generate_report(risk_df)
        
        print("\nPipeline completed successfully!")
        return risk_df, report


if __name__ == "__main__":
    pipeline = TelecomChurnPipeline()
    risk_df, report = pipeline.run_pipeline()
