# model_training.py

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load processed data
X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").values.ravel()
y_test = pd.read_csv("data/y_test.csv").values.ravel()

# Initialize models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    "LightGBM": LGBMClassifier()
}

best_accuracy = 0
best_model = None
best_model_name = ""

# Train and evaluate all models
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n🔍 {name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name

# Save the best model
joblib.dump(best_model, "models/best_model.pkl")

print(f"\n✅ Best model: {best_model_name} with accuracy: {best_accuracy:.4f}")
