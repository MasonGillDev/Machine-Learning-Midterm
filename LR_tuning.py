from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd

# Loadsß dataset
data_path = "./filtered_data.csv"
df = pd.read_csv(data_path)

# Encode cancer types to integers since datapoints starts with cancer type
df["Unnamed: 0"] = df["Unnamed: 0"].apply(lambda x: 1 
    if x.startswith('brca') 
    else (2 if x.startswith('prad') 
    else (3 if x.startswith('luad') else x)))

# Split X and Y into features and target variable
X = df.drop(columns=["Unnamed: 0"])
y = df["Unnamed: 0"]

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (important for logistic regression with high scale features) - Not needed here
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()

# Define parameter grid
param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet', 'None'],  
    'C': [0.01, 0.1],  
    'solver': ['lbfgs', 'saga', 'newton_cholesky'],  
    'max_iter': [100, 1000],  
    'class_weight': [None, 'balanced']
}


# Define grid search model
grid_search = GridSearchCV(model, param_grid, scoring='accuracy', n_jobs=-1, verbose=3, cv=[(slice(None), slice(None))])

# Training
grid_search.fit(X_train, y_train)

# Best parameter output
print("Best Parameters:", grid_search.best_params_)
print(grid_search.best_score_)

# Best Parameters: {'C': 0.1, 'class_weight': 'balanced', 'max_iter': 100, 'penalty': 'l2', 'solver': 'liblinear'}