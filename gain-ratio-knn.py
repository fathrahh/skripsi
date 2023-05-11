import pandas as pd
import pickle

from numpy import mean

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, RepeatedStratifiedKFold, cross_validate

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

# Remove Other value in gender column
df = df[df['gender'] != 'Other']

# Label Encoding
df['gender'] = df['gender'].replace(num_gender)
df['ever_married'] = df['ever_married'].replace(num_ever_married)
df['Residence_type'] = df['Residence_type'].replace(num_residence_type)
df['smoking_status'] = df['smoking_status'].replace(num_smoking_status)
df['work_type'] = df['work_type'].replace(num_work_type)

mean_bmi_replacement_value = df.loc[:, 'bmi'].dropna().mean()

df['bmi'] = df.loc[:, 'bmi'].fillna(mean_bmi_replacement_value)

X = df.iloc[:, :-1].values
y = df.iloc[:, 10].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1)

oversample = SMOTE()

# Resampling

selector = SelectKBest(score_func=mutual_info_classif, k=7)
selector.fit(X_train, y_train)

X_train_new = selector.transform(X_train)
X_test_new = selector.transform(X_test)

X_oversample, y_oversample = oversample.fit_resample(
    X_train_new, y_train.ravel())

knn = KNeighborsClassifier()
# Define parameter grid
param_grid = {'n_neighbors': [3, 5, 7, 9, 11, 13, 15]}

# Define grid search
grid_search = GridSearchCV(knn, param_grid=param_grid, cv=10)

# Fit grid search to training data
grid_search.fit(X_train_new, y_train)

# Get best estimator and evaluate on test data
best_knn = grid_search.best_estimator_
accuracy = best_knn.score(X_test_new, y_test)

print("Accuracy:", accuracy)

exit()
model = KNeighborsClassifier(
    n_neighbors=13, weights='distance', metric='euclidean')

cv = RepeatedStratifiedKFold(n_splits=10, random_state=1)
scores = cross_validate(model, X_oversample, y_oversample,
                        scoring=['accuracy', 'precision', 'recall', 'roc_auc'], cv=cv)

model.fit(X_oversample, y_oversample)

with open('./backend/model.pkl', 'wb') as f:
    pickle.dump(model, f)

for key in scores:
    print(f"{key}: {mean(scores[key])}")
