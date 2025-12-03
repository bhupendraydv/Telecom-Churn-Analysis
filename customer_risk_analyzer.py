"""Customer Risk Analyzer - Score customers by churn risk"""
import pandas as pd, numpy as np, joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

class CustomerRiskAnalyzer:
    def __init__(self, model_path='models/ensemble.pkl'):
        self.model = joblib.load(model_path)
        self.scalers = joblib.load('models/scalers.pkl')
        self.encoders = joblib.load('models/encoders.pkl')
    
    def predict_risk(self, customer_data):
        """Predict churn risk for single customer"""
        processed = customer_data.copy()
        for col, encoder in self.encoders.items():
            if col in processed:
                processed[col] = encoder.transform([processed[col]])[0]
        X = np.array([list(processed.values())])
        X_scaled = self.scalers['features'].transform(X)
        risk_score = self.model.predict_proba(X_scaled)[0][1]
        risk_level = 'HIGH' if risk_score > 0.6 else 'MEDIUM' if risk_score > 0.3 else 'LOW'
        return {'score': round(risk_score, 3), 'level': risk_level, 'prob': f"{risk_score*100:.1f}%"}
    
    def batch_score(self, df):
        """Score all customers in batch"""
        scores = []
        for idx, row in df.iterrows():
            risk = self.predict_risk(row.to_dict())
            risk['customer_id'] = idx
            scores.append(risk)
        return pd.DataFrame(scores)
    
    def segment_customers(self, df):
        """Segment customers by risk level"""
        risks = self.batch_score(df)
        return {
            'high': risks[risks['level'] == 'HIGH'],
            'medium': risks[risks['level'] == 'MEDIUM'],
            'low': risks[risks['level'] == 'LOW']
        }
    
    def get_recommendations(self, risk_level):
        """Get retention recommendations"""
        recommendations = {
            'HIGH': ['🚨 URGENT retention outreach', '💰 Offer discount', '📞 Assign account manager'],
            'MEDIUM': ['⚠️ Monitor closely', '🎯 Personalized offers', '📧 Re-engagement campaigns'],
            'LOW': ['✅ Maintain service', '📈 Upsell premium', '🎊 VIP recognition']
        }
        return recommendations.get(risk_level, [])
