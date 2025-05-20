import pandas as pd
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import dagshub

# Inisialisasi DagsHub untuk MLflow
dagshub.init(repo_owner='zidsw', repo_name='Eksperimen_SML_Zid_Irsyadin', mlflow=True)

def train_basic_model(input_path):
    # Memuat dataset
    df = pd.read_csv(input_path)
    X = df.drop('Air Quality', axis=1)
    y = df['Air Quality']
    
    # Membagi data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Mengatur MLflow
    mlflow.set_experiment("Basic_Model_Zid_Irsyadin")
    mlflow.xgboost.autolog()
    
    with mlflow.start_run(run_name="basic_xgboost"):
        # Melatih model
        model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
        model.fit(X_train, y_train)
        
        # Memprediksi
        y_pred = model.predict(X_test)
        
        # Menghitung metrik
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        
        print(f"Akurasi: {accuracy}")
    
    return model

if __name__ == "__main__":
    input_path = "Membangun_model/pollution_dataset_preprocessed_advance.csv"
    train_basic_model(input_path)