# loan-approval-ml
Machine learning system that predicts loan default risk using Logistic Regression, Decision Tree, and Random Forest. Built with Python &amp; Scikit-learn on a dataset of 58,000+ applicants. University ML project.

---

## 📌 Problem Statement

Banks face significant financial risk when approving loans to high-risk applicants. This project builds a classification model that helps identify applicants likely to default, allowing banks to make smarter, data-driven decisions.

---

## 📂 Dataset

- **Source:** Kaggle — Loan Approval Dataset
- **Size:** 58,645 applicants, 13 features
- **Target:** `loan_status` (0 = Repaid, 1 = Defaulted)
- **Class imbalance:** ~14.2% default rate

### Features Used

| Feature | Type | Description |
|---|---|---|
| `person_age` | Numeric | Age of applicant |
| `person_income` | Numeric | Annual income |
| `person_emp_length` | Numeric | Years of employment |
| `person_home_ownership` | Categorical | RENT / OWN / MORTGAGE / OTHER |
| `loan_amnt` | Numeric | Loan amount requested |
| `loan_intent` | Categorical | Purpose of loan |
| `loan_grade` | Categorical | Bank assigned grade (A–G) |
| `loan_int_rate` | Numeric | Interest rate |
| `loan_percent_income` | Numeric | Loan amount as % of income |
| `cb_person_default_on_file` | Categorical | Previous default history (Y/N) |
| `cb_person_cred_hist_length` | Numeric | Credit history length (years) |

---

## 🔍 Exploratory Data Analysis

Key findings from EDA:

- **Loan Grade** is the strongest predictor — Grade G applicants default ~80% of the time vs Grade A at ~5%
- **Loan percent income** and **interest rate** are most correlated with defaults
- **Renters** default more than homeowners or mortgage holders
- **Previous defaulters** are 3x more likely to default again
- **Debt consolidation & medical** loans have the highest default intent rates

---

## ⚙️ Methodology

1. **Data Loading** — Loaded train/test CSV files
2. **EDA** — Distribution plots, categorical analysis, correlation heatmap
3. **Preprocessing** — Label encoding of categorical features, train/validation split (80/20)
4. **Model Training** — Three models trained with `class_weight='balanced'` to handle imbalance
5. **Evaluation** — Accuracy, Recall, F1 Score compared across models
6. **Prediction** — Best model used to predict on unseen test data

---

## 🤖 Models & Results

| Model | Accuracy | Recall | F1 Score |
|---|---|---|---|
| Logistic Regression | 77.09% | 83.31% | 50.45% |
| Decision Tree | 91.64% | 71.44% | 70.51% |
| **Random Forest** ✅ | **94.95%** | **69.49%** | **79.40%** |

### Why Random Forest?
- Highest accuracy and F1 score
- Handles imbalanced data well with `class_weight='balanced'`
- Naturally captures non-linear relationships between features
- Resistant to overfitting compared to a single Decision Tree

### Note on Recall vs Accuracy
In banking, **Recall** (catching actual defaulters) is critical. Logistic Regression has the highest recall (83%) but poor F1. Random Forest offers the best overall balance with the highest F1 score.

---

## 📊 Key Visualisations

- Numeric feature distributions by loan status
- Default rate by categorical features
- Feature correlation heatmap
- Feature importance chart
- Confusion matrices for all 3 models

---

## 🛠️ Tech Stack

- **Python 3**
- **Pandas & NumPy** — data manipulation
- **Matplotlib & Seaborn** — visualisation
- **Scikit-learn** — machine learning models & evaluation

---

## 🚀 How to Run

1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/loan-approval-ml.git
```

2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

3. Open the notebook
```bash
jupyter notebook loan_approval_pred.ipynb
```

4. Update the dataset path in the loading cell to point to your local CSV files

---

## 📁 Project Structure

```
loan-approval-ml/
│
├── loan_approval_pred.ipynb   # Main notebook
├── README.md                  # Project documentation
└── data/
    ├── train.csv              # Training data
    └── test.csv               # Test data
```

---

## ⚠️ Limitations

- Model depends on historical data and may not generalise to all real-world cases
- Some external factors (economic conditions, employment sector) are not included
- Class imbalance means the model is less confident on minority (default) predictions

---

## 🔮 Future Scope

- Try advanced models like **XGBoost** or **LightGBM**
- Deploy as an interactive **web application** using Flask or Streamlit
- Add **Explainable AI (SHAP values)** to interpret individual predictions
- Add **hyperparameter tuning** with GridSearchCV

---

## 👨‍💻 Author

Made as a university machine learning project.
