import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

# 1. Create Student Dataset
data = {
    "Study_Hours": [2, 3, 4, 5, 6, 7, 8, 1, 2, 5,
                    6, 7, 3, 8, 4, 9, 1, 6, 7, 3],

    "Attendance": [60, 65, 70, 75, 80, 85, 90, 50, 55, 78,
                   82, 88, 68, 95, 72, 92, 45, 85, 90, 62],

    "Pass": [0, 0, 0, 1, 1, 1, 1, 0, 0, 1,
             1, 1, 0, 1, 0, 1, 0, 1, 1, 0]
}

df = pd.DataFrame(data)

# 2. Features and Target
X = df[["Study_Hours", "Attendance"]]
y = df["Pass"]

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Train Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# 6. Probability Prediction
y_prob = model.predict_proba(X_test)[:, 1]

# 7. ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

print("AUC Score:", round(auc, 4))

# 8. Plot ROC Curve
plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()