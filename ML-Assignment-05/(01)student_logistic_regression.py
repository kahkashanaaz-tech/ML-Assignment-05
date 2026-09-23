import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
data = {
    "Study_Hours": [2,3,4,5,6,7,8,1, 2, 5,
                    6, 7, 3, 8, 4, 9, 1, 6, 7, 3],

    "Attendance": [60, 65, 70, 75, 80, 85, 90, 50, 55, 78,
                   82, 88, 68, 95, 72, 92, 45, 85, 90, 62],

    "Pass": [0, 0, 0, 1, 1, 1, 1, 0, 0, 1,
             1, 1, 0, 1, 0, 1, 0, 1, 1, 0]
}

df = pd.DataFrame(data)

print("Student Dataset:")
print(df)
X = df[["Study_Hours", "Attendance"]]
y = df["Pass"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\n--- Logistic Regression Performance ---")

print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score  : {f1_score(y_test, y_pred):.4f}")
