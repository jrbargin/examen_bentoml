import bentoml
from bentoml.io import JSON
from pydantic import BaseModel
from typing import Dict
from datetime import datetime, timedelta
import jwt

# Secret key and algorithm for JWT authentication
JWT_SECRET_KEY = "your_jwt_secret_key_here"
JWT_ALGORITHM = "HS256"

# User credentials for authentication
USERS = {
    "user123": "password123",
    "user456": "password456"
}

# Charger le modèle sauvegardé
model_runner = bentoml.xgboost.get("xgboost_model:xmxvnlvhekovmauu").to_runner()

# Créer le service avec BentoML
svc = bentoml.Service("admission_api_service", runners=[model_runner])

# Modèle pour valider les entrées des endpoints
class AdmissionInput(BaseModel):
    GRE_Score: float
    TOEFL_Score: float
    University_Rating: int
    SOP: float
    LOR: float
    CGPA: float
    Research: int

# Endpoint pour la prédiction
@svc.api(input=JSON(pydantic_model=AdmissionInput), output=JSON())
async def predict(input_data: AdmissionInput) -> Dict[str, float]:
    """
    Endpoint permettant de faire une prédiction à partir des données utilisateur.
    La requête doit inclure un token JWT valide dans les en-têtes.
    """
    # Récupérer le token à partir des en-têtes
    token = svc.context.request.headers.get("Authorization")
    if not token or not token.startswith("Bearer "):
        return {"detail": "Authorization header missing or invalid"}, 401
    token = token.split("Bearer ")[1]

    # Vérification du token JWT
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        return {"detail": "Token has expired"}, 401
    except jwt.InvalidTokenError:
        return {"detail": "Invalid token"}, 401

    # Si le token est valide, effectuer la prédiction
    input_array = [
        input_data.GRE_Score,
        input_data.TOEFL_Score,
        input_data.University_Rating,
        input_data.SOP,
        input_data.LOR,
        input_data.CGPA,
        input_data.Research,
    ]
    prediction = await model_runner.predict.async_run([input_array])
    return {"admission_chance": prediction[0]}

# Endpoint pour le login
@svc.api(input=JSON(), output=JSON())
async def login(credentials: Dict[str, str]) -> Dict[str, str]:
    """
    Endpoint permettant de sécuriser l'accès à l'API avec un système de login basique.
    """
    username = credentials.get("username")
    password = credentials.get("password")

    # Vérification des identifiants
    if username in USERS and USERS[username] == password:
        token = create_jwt_token(username)
        return {"token": token}
    else:
        return {"detail": "Invalid credentials"}, 401

# Fonction pour créer un token JWT
def create_jwt_token(user_id: str) -> str:
    """
    Génère un token JWT avec une expiration d'une heure.
    """
    expiration = datetime.utcnow() + timedelta(hours=1)
    payload = {
        "sub": user_id,
        "exp": expiration
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token