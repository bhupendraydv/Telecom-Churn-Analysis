"""Automated Report Generator for Churn Analysis"""
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from datetime import datetime

class ReportGenerator:
    def __init__(self, title="Telecom Churn Analysis Report"):
        self.title = title
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sns.set_style("whitegrid")
    
    def create_summary(self, churn_rate, total_customers, revenue_impact):
        """Create executive summary"""
        return f"""
╔════════════════════════════════════════════════════════════════╗
║                    EXECUTIVE SUMMARY                           ║
╚════════════════════════════════════════════════════════════════╝

Report Generated: {self.timestamp}

📊 KEY METRICS:
   • Total Customers: {total_customers:,}
   • Current Churn Rate: {churn_rate:.2f}%
   • Monthly Revenue at Risk: ${revenue_impact:,.2f}

🎯 BUSINESS IMPACT:
   • Customers at Risk: {int(total_customers * churn_rate / 100):,}
   • Projected Annual Loss: ${revenue_impact * 12:,.2f}

💡 RECOMMENDATION:
   Implement targeted retention strategies for high-risk segments.
"""
    
    def create_segment_analysis(self, segments):
        """Create segment analysis"""
        report = "\n" + "═" * 60 + "\n"
        report += "CUSTOMER SEGMENTS ANALYSIS\n"
        report += "═" * 60 + "\n"
        for name, count in segments.items():
            report += f"\n{name.upper()}: {count} customers\n"
        return report
    
    def get_recommendations(self):
        """Generate actionable recommendations"""
        return """
╔════════════════════════════════════════════════════════════════╗
║                RECOMMENDED ACTIONS                              ║
╚════════════════════════════════════════════════════════════════╝

HIGH-RISK SEGMENT:
  🚨 URGENT: Priority retention outreach
  💰 Offer discount on next month
  📞 Personal account manager assignment
  📊 Service quality improvement plan

MEDIUM-RISK SEGMENT:
  ⚠️ Monitor closely
  🎯 Personalized service offers
  📧 Re-engagement email campaigns
  💳 Contract upgrade incentives

LOW-RISK SEGMENT:
  ✅ Maintain service quality
  📈 Upsell premium features
  🎊 VIP loyalty recognition
"""
    
    def save_report(self, content, filename='churn_analysis_report.txt'):
        """Save report to file"""
        with open(filename, 'w') as f:
            f.write(content)
        print(f'✓ Report saved: {filename}')
