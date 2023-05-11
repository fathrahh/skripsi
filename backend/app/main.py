import pickle
import numpy as np
from fastapi import FastAPI

from app.request.models import Features
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
def predict(payload: Features) -> int:
    payload = payload.dict()

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    X_dict = {
        'gender': payload['gender'],
        'age': payload['age'],
        'hypertension': payload['hypertension'],
        'heart_disease': payload['heartDisease'],
        'ever_married': payload['everMarried'],
        'work_type': payload['workType'],
        'residence_type': payload['residentType'],
        'avg_glucose_level': payload['avgGlucoseLevel'],
        'bmi': payload['bmi'],
        'smoking_status': payload['smokingStatus']
    }

    X_test = np.array(list(X_dict.values()))
    y_pred = model.predict(X_test.reshape(-1, len(X_test)))

    return y_pred
