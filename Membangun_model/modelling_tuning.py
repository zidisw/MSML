import pandas as pd
import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import dagshub

# Inisialisasi DagsHub untuk MLflow
dagshub.init(repo_owner='zidsw', repo_name='MSML', mlflow=True)

def train_tuned_model(input_path):
    # Memuat dataset
    df = pd.read_csv(input_path)
    X = df.drop('Air Quality', axis=1)
    y = df['Air Quality']
    
    # Membagi data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Mengatur MLflow
    mlflow.set_experiment("Tuned_Model_Zid_Irsyadin")
    
    with mlflow.start_run(run_name="tuned_xgboost"):
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Model terbaik
        best_model = grid_search.best_estimator_
        
        # Memprediksi
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)
        
        # Menghitung metrik
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
        mean_prediction_time = np.mean([best_model.predict(X_test[i:i+1]).shape[0] for i in range(len(X_test))])
        
        # Manual logging
        mlflow.log_param("best_n_estimators", grid_search.best_params_['n_estimators'])
        mlflow.log_param("best_max_depth", grid_search.best_params_['max_depth'])
        mlflow.log_param("best_learning_rate", grid_search.best_params_['learning_rate'])
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision_weighted", precision)
        mlflow.log_metric("recall_weighted", recall)
        mlflow.log_metric("f1_weighted", f1)
        mlflow.log_metric("roc_auc_ovr", roc_auc)
        mlflow.log_metric("mean_prediction_time_ms", mean_prediction_time)
        
        # Menyimpan model
        mlflow.sklearn.log_model(best_model, "xgboost_tuned")
        
        print(f"Akurasi: {accuracy}")
        print(f"Parameter Terbaik: {grid_search.best_params_}")
    
    return best_model

if __name__ == "__main__":
    input_path = "Membangun_model/pollution_dataset_preprocessed_advance.csv"
    train_tuned_model(input_path)