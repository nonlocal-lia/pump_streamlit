# Import dependencies
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

# Load the dataset in a dataframe object
with open('clean_training_values.pickle', 'rb') as file:
    X_train = pickle.load(file)
train_labels = pd.read_csv('training_set_labels.csv')

# Data Preprocessing
y_label = LabelEncoder()
y_train = y_label.fit_transform(train_labels['status_group'])
cat_columns = [col for col in X_train.columns if X_train[col].dtype == 'O']
ct_ord = ColumnTransformer(transformers=[
    ("ord", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),cat_columns)],
    remainder="passthrough")

# Random Forest classifier
pipe_xgb = Pipeline(steps=[
    ("encode", ct_ord),
    ("scale", StandardScaler(with_mean=False)),
    ("model", XGBClassifier(eval_metric='mlogloss',
                          use_label_encoder=False,
                          colsample_bytree=0.3,
                          eta=0.2,
                          max_depth=10,
                          n_estimators=200))])
pipe_xgb.fit(X_train, y_train)

# Save Model
import joblib
joblib.dump(pipe_xgb, 'model.pkl')
print("Model dumped!")

# Saving the data columns from training
model_columns = list(X_train.columns)
joblib.dump(model_columns, 'model_columns.pkl')
joblib.dump(cat_columns, 'cat_columns.pk1')
print("Models columns dumped!")