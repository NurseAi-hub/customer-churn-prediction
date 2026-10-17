import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)
df.head()
df.isnull().sum()

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
print(df['Churn'].head())
print(df['Contract'].value_counts())

df['Contract'] = df['Contract'].map({'One year' : 0, 'Month-to-month' : 1, 'Two year' : 2 })

df['InternetService'] = df['InternetService'].map({'DSL': 0, 'Fiber optic': 1, 'No':2 })
print(df['InternetService'].head())
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)



df_final = pd.get_dummies(df.drop(['customerID'], axis=1), drop_first=True)


X = df_final.drop('Churn', axis=1)
y = df_final['Churn']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

print(f"Final Professional Accuracy: {model.score(X_test_scaled, y_test)*100:.2f}%")





feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.coef_[0]
})


feature_importance = feature_importance.sort_values(by='Importance', ascending=False)


print("Top 10 Factors Increasing Churn:")
print(feature_importance.head(10))


plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'].head(10), feature_importance['Importance'].head(10), color='salmon')
plt.xlabel('Coefficient Value (Strength of influence)')
plt.title('Top 10 Features Driving Customer Churn')
plt.gca().invert_yaxis()
plt.show()

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# 1. Calculate the probabilities (not just 0 or 1)
y_probs = model.predict_proba(X_test_scaled)[:, 1]

# 2. Calculate ROC curve values
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
auc = roc_auc_score(y_test, y_probs)

# 3. Plotting the ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'Logistic Regression (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--') # The "Random Guess" line
plt.xlabel('False Positive Rate (People we wrongly thought would leave)')
plt.ylabel('True Positive Rate (People we correctly caught leaving)')
plt.title('ROC Curve: Model Performance')
plt.legend()
plt.grid(True)
plt.show()

from sklearn.metrics import classification_report, confusion_matrix


y_pred = model.predict(X_test_scaled)


print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("-" * 50)


print("Classification Report:")
print(classification_report(y_test, y_pred))










