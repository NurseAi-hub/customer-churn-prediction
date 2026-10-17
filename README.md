# 📉 Customer Churn Prediction & Health Informatics Framework

## 🎯 Project Overview
This project builds a robust binary classification model to predict customer churn using machine learning. While demonstrated on a standard telecommunications dataset (Telco Churn), the underlying predictive pipeline shares the exact architectural framework used in **health informatics and clinical data science** to predict **patient non-adherence, chronic care program dropouts, and loss to follow-up**. 

By identifying high-risk individuals proactively, organizations can implement targeted interventions to retain engagement and mitigate negative outcomes.

---

## ⚖️ Model Performance & Key Metrics
* **Accuracy:** 81.76%
* **ROC-AUC Score:** 0.86
* **Primary Model:** Logistic Regression (Optimized with Feature Scaling & Categorical Encoding)

---

## 🧩 Confusion Matrix & Evaluation Suite
To move beyond basic accuracy and evaluate true operational risk, the model was tested using a comprehensive classification suite:
* **Confusion Matrix:** Evaluated the balance between correctly identified departures versus false alarms. In a clinical context, this metric translates directly to managing the trade-off between false positives (over-allocating care coordination resources) and false negatives (failing to catch patients dropping out of critical treatment plans).
* **ROC Curve (AUC = 0.86):** Confirms strong discriminatory capacity across varying decision thresholds, ensuring reliable risk stratification.

---

## 🛠️ Data Preprocessing & Pipeline Engineering
To ensure scientific rigor and prevent data leakage:
1. **Data Cleaning:** Converted `TotalCharges` to numeric format and handled missing values cleanly.
2. **Categorical Encoding:** Applied One-Hot Encoding (`pd.get_dummies` with `drop_first=True`) to convert multi-class text features into numerical variables while preventing multicollinearity.
3. **Feature Scaling:** Implemented `StandardScaler` fitted **strictly on training data** (`X_train`) before transforming the test set, maintaining strict separation to eliminate data leakage.

---

## 💡 Key Insights & Feature Importance
Model coefficient analysis revealed the primary drivers influencing customer retention:
* **Contract Commitment:** Month-to-month terms present the highest risk of termination.
* **Service Tier:** Specific service attributes (such as fiber optic usage) showed higher churn correlation, pointing toward pricing or service-experience sensitivities.
* **Tenure:** Long-term engagement acts as a strong protective factor against churn.

---

## 🏥 Clinical & Public Health Application
> **Translating Telecom Churn to Health Systems:** 
> In public health and hospital management, "churn" or cancellation mirrors **patient dropouts from chronic disease management programs** (e.g., diabetes or hypertension follow-ups). Predicting who is likely to disengage allows care coordinators to intervene early, preventing emergency readmissions and improving long-term health outcomes.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (`LogisticRegression`, `StandardScaler`, `train_test_split`, metrics)
* **Visualization:** Matplotlib, Seaborn (`roc_curve`, `confusion_matrix`)
