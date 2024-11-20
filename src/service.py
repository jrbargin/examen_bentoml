import joblib

best_model_loaded = joblib.load('../models/best_model.pkl')
print("Modèle chargé avec joblib")