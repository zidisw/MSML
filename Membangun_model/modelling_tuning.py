import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, log_loss, roc_auc_score, classification_report
)
import os
import json

# Load dataset
df = pd.read_csv("Membangun_model/pollution_dataset_preprocessed_advance.csv")
X = df.drop(columns="Air Quality")
y = df["Air Quality"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 10]
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)

# Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro')
rec = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
logloss = log_loss(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')

params = grid.best_params_
metrics = {
    "training_accuracy_score": acc,
    "training_precision_score": prec,
    "training_recall_score": rec,
    "training_f1_score": f1,
    "training_roc_auc_score": roc_auc,
    "training_log_loss": logloss
}

input_example = X_test.iloc[:1]
signature = infer_signature(X_test, y_pred)

# Simpan artefak tambahan
pd.DataFrame(y_pred, columns=["y_pred"]).to_csv("predictions.csv", index=False)
with open("classification_report.json", "w") as f:
    json.dump(classification_report(y_test, y_pred, output_dict=True), f)

# 1. LOG TO LOCAL (Skilled)
mlflow.set_tracking_uri("")  # local
mlflow.set_experiment("MSML-Skilled-Local")

with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.log_artifact("predictions.csv")
    mlflow.log_artifact("classification_report.json")
    mlflow.sklearn.log_model(
        best_model,
        "best_rf_model",
        signature=signature,
        input_example=input_example
    )

# 2. LOG TO DAGSHUB (Advance)
os.environ["MLFLOW_TRACKING_USERNAME"] = "zidisw"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "78af145a93f6cda50d1106737e56bc4e698b5825"
mlflow.set_tracking_uri("https://dagshub.com/zidisw/Eksperimen_SML_Zid_Irsyadin.mlflow")
mlflow.set_experiment("MSML-Advance-Version")

with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.log_artifact("predictions.csv")
    mlflow.log_artifact("classification_report.json")
    mlflow.sklearn.log_model(
        best_model,
        "best_rf_model",
        registered_model_name="RandomForestModelTuned",
        signature=signature,
        input_example=input_example
    )
