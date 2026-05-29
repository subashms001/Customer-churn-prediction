```markdown
# 🔮 Customer Churn Prediction

Predict which telecom customers are at risk of churning using machine learning.

**Live App:** [Click to try the predictor →](YOUR_STREAMLIT_URL)

## Problem Statement
A telecom company loses revenue every time a customer cancels their subscription.
This project builds a model to identify high-risk customers before they churn,
so the retention team can intervene proactively.

## Dataset
- Source: Kaggle — Telco Customer Churn
- Size: 7,043 customers, 21 features
- Target: Churn (Yes/No) — 26.5% positive rate

## Key Findings (EDA)
- Month-to-month customers churn at 43% vs 3% for two-year contracts
- New customers (< 12 months) are the most vulnerable segment
- Customers without OnlineSecurity churn 3x more than protected customers
- Fiber optic users churn at 42%, nearly double the overall average

## Models Compared

| Model               | Accuracy | Recall (Churn) | AUC-ROC |
|---------------------|----------|----------------|---------|
| Logistic Regression | 76%      | 75%            | 0.83    |
| Random Forest       | 80%      | 74%            | 0.84    |
| XGBoost ✅          | 81%      | 78%            | 0.86    |

XGBoost selected as final model based on highest AUC-ROC and churn recall.

## Top Churn Drivers (SHAP)
1. Contract type (Month-to-month = highest risk)
2. Tenure (new customers are most at risk)
3. Monthly charges (higher bills = more likely to leave)
4. Lack of OnlineSecurity or TechSupport

## Tech Stack
Python · Pandas · Scikit-learn · XGBoost · SHAP · Streamlit

## Project Structure
\`\`\`
churn-prediction/
├── notebooks/     ← EDA, preprocessing, modelling
├── data/          ← dataset and saved charts
├── app/           ← Streamlit app + saved model
└── requirements.txt
\`\`\`

## Run Locally
\`\`\`bash
pip install -r requirements.txt
streamlit run app/app.py
\`\`\`
```

---
---

# Quick Reference — Metrics Explained

| Metric | What it means | Target for this project |
|--------|--------------|------------------------|
| **Accuracy** | Overall correct predictions | Not the main metric here |
| **Precision** | When we predict churn, how often right? | > 0.55 |
| **Recall** | Of all actual churners, how many caught? | > 0.75 ← most important |
| **F1-Score** | Balance of precision and recall | > 0.65 |
| **AUC-ROC** | Overall discrimination ability (0.5–1.0) | > 0.84 |

**Why Recall matters most here:**
Missing a churner (false negative) = customer leaves = lost revenue forever.
Wrongly flagging a loyal customer (false positive) = you call them = small cost.
So catching churners (high recall) is more important than being perfectly precise.
