import requests
from joblib import load
from supabase import create_client
from io import BytesIO
from flask import jsonify

import numpy as np

url = ""
key = ""
# headers = {"apiKey": key, "Authorization": f"Bearer {key}"}

# pass in is_async=True to create an async client
supabase = create_client(url, key)

url_joblib = supabase.storage.from_('model').get_public_url("model.joblib")

response = requests.get(url_joblib)
file_content = response.content

model = load(BytesIO(file_content))

payload = {
    "residentType": 1,
    "smokingStatus": 1,
    "everMarried": 0,
    "workType": 3,
    "gender": 1,
    "heartDisease": 0,
    "hypertension": 1,
    "age": 20,
    "bmi": 20,
    "avgGlucoseLevel": 20
}

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

print(y_pred[0])
