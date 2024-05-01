import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib

df = pd.read_csv("Churn_Modelling.csv")

df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

X = df.drop('Exited', axis=1)
y = df['Exited']
print(X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['Geography', 'Gender']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'RandomForestClassifier': RandomForestClassifier(random_state=42),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'SVC': SVC(probability=True, random_state=42)
}

param_grid = {
    'LogisticRegression': {'classifier__C': [0.1, 1.0, 10]},
    'RandomForestClassifier': {'classifier__n_estimators': [100, 200], 'classifier__max_depth': [5, 10]},
    'KNeighborsClassifier': {'classifier__n_neighbors': [3, 5, 7]},
    'SVC': {'classifier__C': [0.1, 1.0, 10], 'classifier__gamma': ['scale', 'auto']}
}

for model_name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                            #    ('feature_selection', SelectFromModel(estimator=RandomForestClassifier())),
                               ('classifier', model)])
    
    grid_search = GridSearchCV(pipeline, param_grid=param_grid[model_name], cv=5, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    print(f"{model_name} best params: {grid_search.best_params_}")
    print(f"{model_name} best score: {grid_search.best_score_}")
    
    joblib.dump(grid_search.best_estimator_, f'{model_name}_pipeline.pkl')

print("Optimized models and preprocessing pipelines have been saved.")