from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

from numpy import mean

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

filename = 'healthcare-dataset-stroke-data.csv'
cols = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type',
        'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status', 'stroke']

df = pd.read_csv(filename, usecols=cols)

non_numeric_col = ['gender', 'ever_married',
                   'work_type', 'Residence_type', 'smoking_status']


# Data cleansing
modus_smoking_status = df['smoking_status'].max()
df['smoking_status'] = df['smoking_status'].apply(
    lambda x: modus_smoking_status if x == 'Unknown' else x)

mean_bmi_replacement_value = df.loc[:, 'bmi'].dropna().mean()
df['bmi'] = df.loc[:, 'bmi'].fillna(mean_bmi_replacement_value)

df = df[df['gender'] != 'Other']

num_gender = {'Female': 0, 'Male': 1, 'Other': 2}
num_ever_married = {'No': 0, 'Yes': 1}
num_smoking_status = {
    'formerly smoked': 0,
    'never smoked': 1,
    'smokes': 2,
    'Unknown': 3,
}
num_work_type = {
    'children': 0,
    'Govt_job': 1,
    'Never_worked': 2,
    'Private': 3,
    'Self-employed': 4,
}
num_residence_type = {
    'Urban': 0,
    'Rural': 1
}

df['gender'] = df['gender'].replace(num_gender)
df['ever_married'] = df['ever_married'].replace(num_ever_married)
df['Residence_type'] = df['Residence_type'].replace(num_residence_type)
df['smoking_status'] = df['smoking_status'].replace(num_smoking_status)
df['work_type'] = df['work_type'].replace(num_work_type)


kf = KFold(n_splits=10, random_state=1, shuffle=True)
scores = {
    'fmeasure': []
}

column = ["age", "avg_glucose_level", "bmi"]

scaler = MinMaxScaler()

for col in cols:
    df[col] = scaler.fit_transform(df[[col]])

X = df.iloc[:, df.columns != 'stroke'].values
y = np.array(df['stroke'])

model = KNeighborsClassifier(
    n_neighbors=3, weights='distance', metric='euclidean')
oversampler = SMOTE(random_state=42)
tl = TomekLinks()

# Definisikan pipeline dengan langkah-langkah pemrosesan data dan model
pipeline = Pipeline(steps=[
    ('oversampling', oversampler),
    ('tomek', tl),
    ('classification', model)
])

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    scores['fmeasure'].append(f1_score(y_test, y_pred))

for key in scores:
    print(f"{key}: {mean(scores[key])}")
