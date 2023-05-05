from fastapi import FastAPI
from starlette.responses import Response

from .request.models import Features
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "Fast API in Python"}


@app.post("/predict", status_code=200)
def predict(payload: Features):
    payload = payload.dict()

    return payload['age']
