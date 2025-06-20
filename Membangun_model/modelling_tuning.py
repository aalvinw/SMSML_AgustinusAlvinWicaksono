import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import mlflow
import mlflow.sklearn
import dagshub
import os 

# Inisialisasi DagsHub
os.environ["MLFLOW_TRACKING_USERNAME"] = "aalvinw"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "fd0f43f8991441bdbdfa6807f7e71cf33c2b9082"  # <- token asli kamu
mlflow.set_tracking_uri("https://dagshub.com/aalvinw/submission-msml_Agustinusalvinwicaksono.mlflow")
# Load data hasil preprocessing
data = pd.read_csv(r"C:\Users\ASUS\Downloads\SMSML_AgustinusAlvinWicaksono\SMSML_AgustinusAlvinWicaksono\Membangun_model\preprocessed_data.csv")
X = data.drop("Reached.on.Time_Y.N", axis=1)
y = data["Reached.on.Time_Y.N"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Parameter untuk GridSearch
params = {
    "n_estimators": [100, 150],
    "max_depth": [5, 10]
}

with mlflow.start_run(run_name="GridSearch_RF"):
    clf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid=params, cv=3)
    clf.fit(X_train, y_train)

    best_model = clf.best_estimator_
    y_pred = best_model.predict(X_test)

    # Logging parameter terbaik
    mlflow.log_param("best_params", clf.best_params_)
    mlflow.log_param("model_type", "RandomForestClassifier")

    # Logging metrik manual
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision_macro", report["macro avg"]["precision"])
    mlflow.log_metric("recall_macro", report["macro avg"]["recall"])

    # Logging model ke MLflow
    mlflow.sklearn.log_model(
    sk_model=best_model,
    artifact_path="model",
    registered_model_name=None
)


print("✅ Model training dan logging ke MLflow selesai.")
