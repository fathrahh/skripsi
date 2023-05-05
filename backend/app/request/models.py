from pydantic import BaseModel

class Features(BaseModel):
    age: int
    BMI: float
    avgGlucoseLevel: float
    gender: int
    hypertension: int
    heartDisease: int
    everMarried: int
    workType: int
    residentType: int
    smokingStatus: int
