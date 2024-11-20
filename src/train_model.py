from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import pandas as pd
import joblib
import bentoml

# Chargement des jeux de données
X_train = pd.read_csv("../data/processed/X_train.csv")
X_test = pd.read_csv("../data/processed/X_test.csv")
y_train = pd.read_csv("../data/processed/y_train.csv")
y_test = pd.read_csv("../data/processed/y_test.csv")

# Initialisation du modèle XGBRegressor
model_xgb = XGBRegressor()

# Définition des paramètres pour le GridSearch
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 500, 1000],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Recherche des meilleurs paramètres avec GridSearchCV
grid_search = GridSearchCV(estimator=model_xgb, param_grid=param_grid, cv=5, scoring='r2', verbose=1)
grid_search.fit(X_train, y_train)

# Affichage des meilleurs paramètres trouvés
print("Best parameters found by GridSearchCV: ", grid_search.best_params_)

# Récupération du meilleur modèle
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Affichage des scores sur les données d'entraînement et de test
print("Score train (R2): {:.2%}".format(best_model.score(X_train, y_train)))
print("Score test (R2): {:.2%}".format(best_model.score(X_test, y_test)))

# Sauvegarde du modèle avec joblib
joblib.dump(best_model, '../models/best_model.pkl')
print("Modèle sauvegardé dans 'models/best_model.pkl'")

# Enregistrement du modèle dans BentoML
model_ref = bentoml.xgboost.save_model("xgboost_model", best_model)
print(f"Modèle enregistré dans BentoML sous : {model_ref}")
