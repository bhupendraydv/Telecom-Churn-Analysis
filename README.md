# Telecom Customer Churn Analysis

## Project Overview

This project provides a **comprehensive machine learning analysis** to predict customer churn in the telecom industry. Using advanced data science techniques, we analyze customer behavior patterns and build predictive models to identify high-risk customers.

### Business Context
- **Churn Rate**: ~27% of customers in the dataset
- **Objective**: Predict which customers are likely to churn
- **Impact**: Reduce customer acquisition costs by focusing on retention

---

## 📊 Dataset Information

**File**: `TelecomCustomerChurn.csv`
- **Samples**: 7,043 customer records
- **Features**: 20 customer attributes
- **Target**: Churn (Yes/No)

### Key Attributes
- Customer demographics (gender, age group)
- Account information (tenure, contract type)
- Service usage (internet, phone, streaming)
- Billing details (charges, payment method)

---

## 🔍 Data Analysis Workflow

### 1. **Exploratory Data Analysis (EDA)**
- Statistical summary of features
- Churn distribution analysis
- Visualization of key patterns
- Correlation analysis

### 2. **Data Preprocessing**
- Handling missing values
- Encoding categorical variables
- Feature scaling using StandardScaler
- Train-test split (80-20)

### 3. **Feature Engineering**
- Numerical and categorical feature extraction
- Feature importance ranking
- Dimensionality analysis

### 4. **Model Development**
Multiple ML models implemented:

#### Logistic Regression
- Linear baseline model
- Probability interpretation
- Feature coefficients analysis

#### Random Forest Classifier
- Ensemble method
- Feature importance extraction
- Non-linear pattern detection

### 5. **Model Evaluation**
- **Accuracy**: Overall correctness
- **Precision & Recall**: For imbalanced classes
- **ROC-AUC Score**: Model discrimination ability
- **Confusion Matrix**: Detailed performance breakdown

---

## 📁 Project Structure

```
Telecom-Churn-Analysis/
├── README.md
├── telecom_churn_analysis.py    # Main analysis script
├── requirements.txt              # Dependencies
├── TelecomCustomerChurn.csv     # Dataset
└── notebooks/
    └── analysis_notebook.ipynb  # Detailed analysis
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Pip package manager

### Installation

```bash
# Clone repository
git clone https://github.com/bhupendraydv/Telecom-Churn-Analysis.git
cd Telecom-Churn-Analysis

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis

```bash
# Execute main analysis
python telecom_churn_analysis.py

# Or use Jupyter
jupyter notebook notebooks/analysis_notebook.ipynb
```

---

## 📈 Key Findings

### Churn Patterns
- Customers with month-to-month contracts have higher churn
- Fiber optic internet users show higher churn rates
- Longer tenure correlates with lower churn
- Senior citizens have slightly higher churn

### Top Predictive Features
1. Contract type
2. Internet service type
3. Monthly charges
4. Tenure
5. Total charges

### Model Performance
- **Random Forest Accuracy**: ~80%
- **ROC-AUC Score**: ~0.84
- **Best for identifying at-risk customers**

---

## 💡 Business Recommendations

1. **Focus on Retention**: Prioritize customers with month-to-month contracts
2. **Service Improvements**: Address fiber optic service quality issues
3. **Loyalty Programs**: Incentivize long-term contracts
4. **Personalized Offers**: Target high-risk segments with retention offers

---

## 📚 Technologies Used

- **Python 3.8+**
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning
- **Matplotlib & Seaborn**: Visualization
- **XGBoost**: Advanced modeling
- **Jupyter**: Interactive notebooks

---

## 📊 Performance Metrics

| Model | Accuracy | Precision | Recall | ROC-AUC |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.78 | 0.65 | 0.52 | 0.82 |
| Random Forest | 0.82 | 0.71 | 0.61 | 0.84 |

---

## 🔗 Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas User Guide](https://pandas.pydata.org/docs/)
- [Machine Learning Mastery](https://machinelearningmastery.com/)

---

## 👤 Author

**Bhupendra Yadav**
- VOIS AI/ML Bootcamp
- GitHub: [@bhupendraydv](https://github.com/bhupendraydv)

---

## 📄 License

This project is open source and available under the MIT License.

---

## 📞 Contact & Support

For questions or suggestions, please open an issue on GitHub or contact the author.

---

**Last Updated**: November 2025
