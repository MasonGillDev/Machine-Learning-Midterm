import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data_path = "./filtered_data.csv"
df = pd.read_csv(data_path)

# Encode cancer types to integers since datapoints starts with cancer type
df["Unnamed: 0"] = df["Unnamed: 0"].apply(lambda x: 1 
    if x.startswith('brca') 
    else (2 if x.startswith('prad') 
    else (3 if x.startswith('luad') else x)))

# Split into features and target variable
X = df.drop(columns=["Unnamed: 0"])
y = df["Unnamed: 0"]

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (important for logistic regression with high scale features) - Not needed here
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Logistic Regression model with max iterations
model = LogisticRegression(C=0.1, class_weight=None, max_iter=100, penalty='l2', solver='lbfgs')

# Train model
model.fit(X_train, y_train)
# model.fit(X_train_scaled, y_train)

# Make prediction
y_pred = model.predict(X_test)
#y_pred = model.predict(X_test_scaled)

# Evaluate the model and print report
accuracy = (accuracy_score(y_test, y_pred)*100)

print(f'\nAccuracy: {accuracy:.2f}%')

print('\nClassification Report:')
print(classification_report(y_test, y_pred))

#Best Parameters: {'C': 0.1, 'class_weight': None, 'max_iter': 100, 'penalty': 'l2', 'solver': 'lbfgs'}