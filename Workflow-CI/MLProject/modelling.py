import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Aktifkan autolog MLflow
mlflow.autolog()

# 1. Load data preprocessed
df = pd.read_csv("https://raw.githubusercontent.com/aalvinw/SMSML_AgustinusAlvinWicaksono/refs/heads/main/Workflow-CI/MLProject/preprocessed_data.csv")

# 2. Pisahkan fitur dan target
X = df.drop(columns=["Reached.on.Time_Y.N"])
y = df["Reached.on.Time_Y.N"]

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ❌ Tidak perlu set_experiment atau start_run di sini

# 4. Model sederhana
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# 5. Evaluasi
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Akurasi:", acc)
print(classification_report(y_test, y_pred))

# ✅ MLflow akan log semua secara otomatis (param, metric, model, dll)
