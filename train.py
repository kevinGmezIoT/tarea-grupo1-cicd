import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from model_utils import save_model_bundle
import os

def load_and_preprocess_data():
    # Load data
    data_path = os.path.join("data", "diabetes.csv")
    df = pd.read_csv(data_path)
    
    # Handle zeros in specific columns
    zero_as_nan_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_as_nan_cols:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())
    
    # Split features and target
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Balance classes using SMOTE
    sm = SMOTE(random_state=42)
    X_bal, y_bal = sm.fit_resample(X_scaled, y)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.25, random_state=42, stratify=y_bal
    )
    
    return X_train, X_test, y_train, y_test, X.columns.tolist()

def train_and_eval():
    # Load and preprocess data
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data()
    
    # Train model
    clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba))
    }
    
    # Save model bundle
    bundle = {
        "model": clf,
        "feature_names": feature_names,
        "target_names": ["No Diabetes", "Diabetes"]
    }
    save_model_bundle(bundle)
    
    print("Training completed. Metrics:", metrics)
    return metrics

if __name__ == "__main__":
    train_and_eval()
