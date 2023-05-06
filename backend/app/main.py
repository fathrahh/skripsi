from fastapi import FastAPI, Request

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

    print(payload)

    return payload['age']
