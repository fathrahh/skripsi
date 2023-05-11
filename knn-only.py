import pandas as pd
import numpy as np
import pickle

from numpy import mean
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold

# main
filename = 'healthcare-dataset-stroke-data.csv'
cols = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type',
        'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status', 'stroke']

df = pd.read_csv(filename, usecols=cols)

# Data cleansing
num_gender = {'Female': 0, 'Male': 1}
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

df['gender'] = df[['gender' != 'other']]

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

n_folds = 10

# Initialize the KFold object
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

model = KNeighborsClassifier(n_neighbors=3, weights="distance")

accuracy_scores = []
predicted_labels = []

for train_index, test_index in kf.split(X):
    # Split the data into training and testing sets for this fold
    print(train_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Evaluate the model on the testing data and store the accuracy score
    accuracy_scores.append(model.score(X_test, y_test))

# Calculate the mean and standard deviation of the accuracy scores
mean_accuracy = np.mean(accuracy_scores)
std_accuracy = np.std(accuracy_scores)

print(f'Mean accuracy: {mean_accuracy:.f} +/- {std_accuracy:.3f}')
