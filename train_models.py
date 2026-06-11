import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

# Create models directory
os.makedirs("models", exist_ok=True)

def train_spectral_models():
    """
    Trains classification and regression models using hyperspectral data.
    """
    print("--- Training Spectral/NIR Models ---")
    if not os.path.exists("data/spectral_dataset.csv"):
        raise FileNotFoundError("Run generate_data.py first to create the dataset.")
        
    df = pd.read_csv("data/spectral_dataset.csv")
    
    # Encode categorical features
    fruit_encoder = LabelEncoder()
    df['fruit_type_encoded'] = fruit_encoder.fit_transform(df['fruit_type'])
    
    # We will save the encoder
    joblib.dump(fruit_encoder, "models/fruit_encoder.joblib")
    
    # Feature columns (fruit_type_encoded + 100 spectral bands)
    band_cols = [f"band_{i+1}" for i in range(100)]
    X = df[['fruit_type_encoded'] + band_cols]
    
    # 1. Train Freshness Classifier (Fresh, Moderately Fresh, Spoiled)
    y_class = df['freshness']
    
    # Encode freshness labels
    freshness_encoder = LabelEncoder()
    y_class_encoded = freshness_encoder.fit_transform(y_class)
    joblib.dump(freshness_encoder, "models/freshness_encoder.joblib")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_class_encoded, test_size=0.2, random_state=42, stratify=y_class_encoded)
    
    print("Training Spectral Classifier (Random Forest)...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Spectral Classifier Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=freshness_encoder.classes_))
    
    # Save Spectral Classifier
    joblib.dump(clf, "models/spectral_classifier.joblib")
    
    # 2. Train Chemical Regressors (Brix, Moisture, Firmness)
    print("Training Chemical Regressors...")
    chemical_targets = ['sugar_brix', 'moisture_pct', 'firmness_n']
    regressors = {}
    reg_metrics = {}
    
    for target in chemical_targets:
        y_reg = df[target]
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)
        
        reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        reg.fit(X_train_r, y_train_r)
        
        y_pred_r = reg.predict(X_test_r)
        r2 = r2_score(y_test_r, y_pred_r)
        rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
        print(f"Spectral Regressor [{target}] - R2: {r2:.4f}, RMSE: {rmse:.4f}")
        
        # Save Regressor
        joblib.dump(reg, f"models/{target}_regressor.joblib")
        reg_metrics[target] = {"R2": r2, "RMSE": rmse}
        
    return {
        "spectral_classifier_accuracy": acc,
        "spectral_regressor_metrics": reg_metrics
    }

def train_enose_models():
    """
    Trains classification models using E-Nose gas and environmental sensor readings.
    """
    print("\n--- Training E-Nose & Environmental IoT Models ---")
    if not os.path.exists("data/sensor_dataset.csv"):
        raise FileNotFoundError("Run generate_data.py first to create the dataset.")
        
    df = pd.read_csv("data/sensor_dataset.csv")
    
    # Encode categorical features
    fruit_encoder = joblib.load("models/fruit_encoder.joblib")
    df['fruit_type_encoded'] = fruit_encoder.transform(df['fruit_type'])
    
    # Features: fruit_type_encoded, temp_c, humidity_pct, gas readings
    feature_cols = [
        'fruit_type_encoded', 'temp_c', 'humidity_pct',
        'mq135_co2_ppm', 'mq3_alcohol_ppm', 'mq4_methane_ppm',
        'mq137_ammonia_ppm', 'mq138_benzene_ppm'
    ]
    
    X = df[feature_cols]
    y_class = df['freshness']
    
    freshness_encoder = joblib.load("models/freshness_encoder.joblib")
    y_class_encoded = freshness_encoder.transform(y_class)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_class_encoded, test_size=0.2, random_state=42, stratify=y_class_encoded)
    
    # Scale E-Nose features because sensor readings have different scales (ppm ranges)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save Scaler
    joblib.dump(scaler, "models/enose_scaler.joblib")
    
    # Train SVM classifier
    print("Training E-Nose Classifier (Support Vector Machine)...")
    svm_clf = SVC(probability=True, random_state=42)
    svm_clf.fit(X_train_scaled, y_train)
    svm_y_pred = svm_clf.predict(X_test_scaled)
    svm_acc = accuracy_score(y_test, svm_y_pred)
    print(f"SVM E-Nose Classifier Accuracy: {svm_acc*100:.2f}%")
    
    # Train Gradient Boosting classifier
    print("Training E-Nose Classifier (Gradient Boosting)...")
    gb_clf = GradientBoostingClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    gb_clf.fit(X_train, y_train)
    gb_y_pred = gb_clf.predict(X_test)
    gb_acc = accuracy_score(y_test, gb_y_pred)
    print(f"Gradient Boosting E-Nose Classifier Accuracy: {gb_acc*100:.2f}%")
    print(classification_report(y_test, gb_y_pred, target_names=freshness_encoder.classes_))
    
    # Save Gradient Boosting as the main E-Nose model
    joblib.dump(gb_clf, "models/enose_classifier.joblib")
    
    # Train Spoilage Index Regressor (based on environmental conditions)
    # Predicts general spoilage risk (0-1) from sensors
    print("Training Spoilage Index Regressor...")
    y_reg = df['spoilage_index']
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    
    spoilage_reg = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    spoilage_reg.fit(X_train_r, y_train_r)
    
    sp_pred = spoilage_reg.predict(X_test_r)
    sp_r2 = r2_score(y_test_r, sp_pred)
    print(f"Spoilage Regressor R2: {sp_r2:.4f}")
    
    joblib.dump(spoilage_reg, "models/spoilage_regressor.joblib")
    
    return {
        "enose_svm_accuracy": svm_acc,
        "enose_gb_accuracy": gb_acc,
        "spoilage_regressor_r2": sp_r2
    }

if __name__ == "__main__":
    spec_metrics = train_spectral_models()
    sensor_metrics = train_enose_models()
    
    metrics = {**spec_metrics, **sensor_metrics}
    
    with open("models/evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("\n--- Training Complete! Model performance metrics saved. ---")
