import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# 1. Create a fresh models directory
os.makedirs("models", exist_ok=True)

print("⏳ Loading data...")
df = pd.read_csv("data/Synthetic_patient-HealthCare-Monitoring_dataset.csv")

if 'Patient Number' in df.columns:
    df.drop(columns=['Patient Number'], inplace=True)

le_features = LabelEncoder()
le_target = LabelEncoder()

cat_cols = ['Fall Detection', 'Heart Rate Alert', 'SpO2 Level Alert', 'Blood Pressure Alert', 'Temperature Alert']
for col in cat_cols:
    df[col] = le_features.fit_transform(df[col])

X = df.drop('Predicted Disease', axis=1)
y = le_target.fit_transform(df['Predicted Disease'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("⚙️ Training Random Forest model (this might take a few seconds)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Using optimized parameters
rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)

print("💾 Saving files...")
model_path = "models/physio_monitor_rf_optimized.pkl"
scaler_path = "models/feature_scaler.pkl"
target_path = "models/le_target.pkl"

joblib.dump(rf, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(le_target, target_path)

# 2. Verify files are not corrupted (Size should be > 0)
print("\n--- File Verification ---")
print(f"Model size: {os.path.getsize(model_path) / 1024:.2f} KB")
print(f"Scaler size: {os.path.getsize(scaler_path) / 1024:.2f} KB")
print(f"Target Encoder size: {os.path.getsize(target_path) / 1024:.2f} KB")
print("✅ Training complete and files saved successfully!")