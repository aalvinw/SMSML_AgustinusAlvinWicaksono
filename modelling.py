import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn

# 1. Load data preprocessed
df = pd.read_csv("https://raw.githubusercontent.com/aalvinw/SMSML_AgustinusAlvinWicaksono/refs/heads/main/Membangun_model/preprocessed_data.csv")

# 2. Pisahkan fitur dan target
X = df.drop(columns=["Reached.on.Time_Y.N"])
y = df["Reached.on.Time_Y.N"]

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# 5. Evaluasi
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Akurasi:", acc)
print(classification_report(y_test, y_pred))

# 6. Logging manual
mlflow.set_experiment("Basic_Model_Logistics")
mlflow.log_param("model_type", "LogisticRegression")
mlflow.log_param("max_iter", 500)
mlflow.log_metric("accuracy", acc)

# 7. Simpan model
mlflow.sklearn.log_model(model, "model")
