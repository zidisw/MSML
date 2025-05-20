import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

# Konfigurasi koneksi ke DagsHub (online atau bisa diganti lokal)
os.environ["MLFLOW_TRACKING_USERNAME"] = "zidisw"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "78af145a93f6cda50d1106737e56bc4e698b5825"
mlflow.set_tracking_uri("https://dagshub.com/zidisw/Eksperimen_SML_Zid_Irsyadin.mlflow")
mlflow.set_experiment("MSML-Basic-Final")

# Aktifkan autolog 
mlflow.sklearn.autolog()

# Load dataset
df = pd.read_csv("Membangun_model/pollution_dataset_preprocessed_advance.csv")
X = df.drop(columns="Air Quality")
y = df["Air Quality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Jalankan run dengan autolog aktif
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
