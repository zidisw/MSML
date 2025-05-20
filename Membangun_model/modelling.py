import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

# Set tracking ke lokal
mlflow.set_tracking_uri("")
mlflow.set_experiment("MSML-Basic-Local")

# Aktifkan autolog (wajib untuk Basic)
mlflow.sklearn.autolog()

# Load data
df = pd.read_csv("Membangun_model/pollution_dataset_preprocessed_advance.csv")
X = df.drop(columns="Air Quality")
y = df["Air Quality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train dan logging
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
