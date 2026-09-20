import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score

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

# 6. Get probability of Pass
probability = model.predict_proba(X_test)[:, 1]

# 7. Test Different Thresholds
for threshold in [0.3, 0.5, 0.7]:

    y_pred = (probability >= threshold).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    print("\nThreshold:", threshold)
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))