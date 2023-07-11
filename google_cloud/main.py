import functions_framework
import numpy as np
import requests
import os
import json
import pandas as pd

from joblib import load
from supabase import create_client
from io import BytesIO


@functions_framework.http
def hello_http(request):
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.
    """
    payload = request.get_json(silent=True)
    request_args = request.args

    url = os.environ.get("supa_url")
    key = os.environ.get("supa_key")

    supabase = create_client(url, key)
    url_joblib = supabase.storage.from_('model').get_public_url("model.joblib")
    url_scaler = supabase.storage.from_(
        'model').get_public_url("scaler.joblib")

    response = requests.get(url_joblib)
    file_content = response.content
    model = load(BytesIO(file_content))

    response = requests.get(url_scaler)
    file_content = response.content
    scaler = load(BytesIO(file_content))

    X_dict = {
        'age': payload['age'],
        'hypertension': payload['hypertension'],
        'ever_married': payload['everMarried'],
        'work_type': payload['workType'],
        'bmi': payload['bmi'],
    }

    df = pd.DataFrame.from_dict(X_dict, orient='index')
    X_test = df.transpose()
    X_test = scaler.transform(X_test)
    print(X_test)
    X_test = pd.DataFrame(X_test, columns=X_dict.keys())
    y_pred = model.predict(X_test)

    if y_pred[0] == 0:
        return 'tidak stroke'

    return 'stroke'
