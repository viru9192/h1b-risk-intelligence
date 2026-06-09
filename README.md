# H1B Employer Risk Intelligence Platform

Predicts H1B visa denial risk for US employers using 187,000+ 
employer records from DOL LCA filings and USCIS petition data.

## 🚀 Live App
👉 [Open Dashboard](https://h1b-risk-intelligence-u6virxq3rxjosrigiadagf.streamlit.app)

## Key Findings
- Professional & Technical Services has highest denial rate at 9.1%
- 2018 peak denial rate of 13.7% driven by USCIS policy changes
- Employers paying below 90% of prevailing wage face 24.5% denial rate
- Paying above 150% of prevailing wage reduces denial risk to 3.5%

## Model Performance
- ROC-AUC: 0.9351
- PR-AUC: 0.7298
- High Risk Recall: 88%

## Tech Stack
Python · XGBoost · SHAP · Streamlit · scikit-learn · MLflow · pandas

## Data Sources
- US Department of Labor — LCA filings (2017–2026)
- USCIS — H1B petition approval/denial data
