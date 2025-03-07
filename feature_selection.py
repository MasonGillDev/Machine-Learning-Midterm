import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns
import sys



feature_count = int(sys.argv[1])

data = pd.read_csv('dataset.csv')


data['Unnamed: 0'] = data['Unnamed: 0'].apply(lambda x: 0 if x.startswith('brca') 
    else (1 if x.startswith('prad') 
        else (2 if x.startswith('luad') else  print("unexpected label"))))

data = data.loc[1 :, (data != 0).any(axis=0)]


X = data.drop(columns=['Unnamed: 0'])

data = data.loc[data.sum(axis=1).sort_values(ascending=False).index]

y = data['Unnamed: 0']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    use_label_encoder=False,
    tree_method='hist'  # Faster for high-dimensional data
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)]
)

importance = model.feature_importances_
feature_importance = pd.DataFrame({
    'Gene': X.columns,
    'Importance': importance
})

feature_importance = feature_importance.sort_values('Importance', ascending=False).head(feature_count)['Gene'].tolist()
new_data = pd.read_csv('dataset.csv')
filtered_data = new_data[['Unnamed: 0'] + feature_importance]

filtered_data.to_csv("filtered_data.csv", index=False)