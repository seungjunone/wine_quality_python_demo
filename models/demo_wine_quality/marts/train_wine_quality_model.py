import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier

def transform_categorical(data):
    categories = (data.dtypes =="object")
    cat_cols = list(categories[categories].index)
    label_encoder = LabelEncoder()
    for col in cat_cols:
        data[col] = label_encoder.fit_transform(data[col])

def scale_numerical(data):
    scaler = MinMaxScaler()
    data[data.columns] = scaler.fit_transform(data[data.columns])


def model(dbt, session):

    dbt.config(
        materialized = 'table',
        packages = ["numpy==1.23.1", "scikit-learn", "pandas"]
        )
    
    source_df = dbt.ref('int_red_white_unioned').to_pandas()
    
    X = source_df.drop("TARGET", axis = 1)
    y = source_df["TARGET"]
     
    transform_categorical(X)
    scale_numerical(X)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
     
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
     
    # train model
    model = RandomForestClassifier()


    # predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
     
    # result
    metric_result = [[
        datetime.now(),
        "wine_quality_model_v1",
        datetime.today().strftime('%Y-%m-%d'),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred),
        accuracy_score(y_test, y_pred)
        ]]
     
    result_df = pd.DataFrame(
        metric_result, 
        columns=['model_id','model_name','model_train_date','precision', 'recall', 'f1', 'accuracy']
    )

    return result_df