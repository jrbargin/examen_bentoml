from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import pandas as pd
import joblib

X_train = pd.read_csv("../data/processed/X_train.csv")
X_test = pd.read_csv("../data/processed/X_test.csv")
y_train = pd.read_csv("../data/processed/y_train.csv")
y_test = pd.read_csv("../data/processed/y_test.csv")

model_xgb = XGBRegressor()

param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 500, 1000],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(estimator=model_xgb, param_grid=param_grid, cv=5, scoring='r2', verbose=1)
grid_search.fit(X_train, y_train)

print("Best parameters found by GridSearchCV: ", grid_search.best_params_)

best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)


print("Score train (R2): {:.2%}".format(best_model.score(X_train, y_train)))
print("Score test (R2): {:.2%}".format(best_model.score(X_test, y_test)))


# Sauvegarder le modèle dans le dossier 'model'
joblib.dump(best_model, 'model/best_model.pkl')