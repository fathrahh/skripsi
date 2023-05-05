import pandas as pd
from numpy import mean
from pandas.core.frame import DataFrame

from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_validate


def printColumnTypes(non_numeric_df: DataFrame, numeric_df: DataFrame) -> None:
    '''separates non-numeric and numeric columns'''

    print("Non-Numeric columns:")
    for col in non_numeric_df:
        print(f"{col}")

    print("Numeric columns:")
    for col in numeric_df:
        print(f"{col}")


def printUniqueValue(cols: list):
    for col in cols:
        print(f"{col}: {df[col].unique()}")


# main
filename = 'healthcare-dataset-stroke-data.csv'
cols = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type',
        'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status', 'stroke']

df = pd.read_csv(filename, usecols=cols)

# Check datatype each column of dataframe
cat_df = df.select_dtypes(include=['object'])
num_df = df.select_dtypes(exclude=['object'])

printColumnTypes(cat_df, num_df)

non_numeric_col = ['gender', 'ever_married',
                   'work_type', 'Residence_type', 'smoking_status']

printUniqueValue(non_numeric_col)

# Data cleansing
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

mean_bmi_replacement_value = df.loc[:, 'bmi'].dropna().mean()

df['bmi'] = df.loc[:, 'bmi'].fillna(mean_bmi_replacement_value)

print(df['stroke'].value_counts() / len(df))

X = df.iloc[:, :-1].values
y = df.iloc[:, 10].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1)

oversample = SMOTE()

# Resampling
X_oversample, y_oversample = oversample.fit_resample(X_train, y_train.ravel())
counter = Counter(y_oversample)


model = KNeighborsClassifier(
    n_neighbors=13, weights='distance', metric='euclidean')

cv = RepeatedStratifiedKFold(n_splits=10, random_state=1)
scores = cross_validate(model, X_oversample, y_oversample,
                        scoring=['accuracy', 'precision', 'recall', 'roc_auc'], cv=cv)

for key in scores:
    print(f"{key}: {mean(scores[key])}")
