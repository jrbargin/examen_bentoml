import bentoml
from bentoml.io import JSON
from pydantic import BaseModel
from typing import Dict
from datetime import datetime, timedelta
import jwt
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from bentoml import Service

# Secret key and algorithm for JWT authentication
JWT_SECRET_KEY = "your_jwt_secret_key_here"
JWT_ALGORITHM = "HS256"

# User credentials for authentication
USERS = {
    "user123": "password123",
    "user456": "password456"
}

class JWTAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.url.path == "/predict":
            token = request.headers.get("Authorization")
            if not token:
                return JSONResponse(status_code=401, content={"detail": "Missing authentication token"})
            
            try:
                token = token.split()[1]  # Retirer le préfixe "Bearer "
                payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            except jwt.ExpiredSignatureError:
                return JSONResponse(status_code=401, content={"detail": "Token has expired"})
            except jwt.InvalidTokenError:
                return JSONResponse(status_code=401, content={"detail": "Invalid token"})
            
            # Ajouter l'utilisateur dans le contexte de la requête
            request.state.user = payload.get("sub")
        
        response = await call_next(request)
        return response

class AdmissionInput(BaseModel):
    GRE_Score: float
    TOEFL_Score: float
    University_Rating: int
    SOP: float
    LOR: float
    CGPA: float
    Research: int

# Get the model from the Model Store
model_runner = bentoml.sklearn.get("xgboost_model:xmxvnlvhekovmauu").to_runner()

# Create a service API
svc = bentoml.Service("admission_api_service", runners=[model_runner])

# Add the JWTAuthMiddleware to the service
svc.add_asgi_middleware(JWTAuthMiddleware)



# Endpoint pour se connecter et obtenir un token
@svc.api(input=JSON(), output=JSON())
def login(credentials: Dict[str, str]) -> Dict[str, str]:
    username = credentials.get("username")
    password = credentials.get("password")

    if username in USERS and USERS[username] == password:
        token = create_jwt_token(username)
        return {"token": token}
    else:
        return JSONResponse(status_code=401, content={"detail": "Invalid credentials"})

# Endpoint pour la prédiction
@svc.api(input=JSON(pydantic_model=AdmissionInput), output=JSON())
async def predict(input_data: AdmissionInput) -> Dict[str, float]:
  
    user = svc.context.request.state.user

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
    return {"admission_chance": prediction[0], "user": user}


# Function to create a JWT token
def create_jwt_token(user_id: str):
    expiration = datetime.utcnow() + timedelta(hours=1)
    payload = {
        "sub": user_id,
        "exp": expiration
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token