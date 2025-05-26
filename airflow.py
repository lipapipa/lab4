import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer # т.н. преобразователь колонок
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import root_mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import requests
from pathlib import Path
import os
from datetime import timedelta
from train_model import train

def download_data():
    df = pd.read_csv('https://raw.githubusercontent.com/Alexey3250/Start_ML/refs/heads/main/Machine_learning/10_classification/banking.csv', delimiter = ',')
    return df

def clear_data():
    df = pd.read_csv('https://raw.githubusercontent.com/Alexey3250/Start_ML/refs/heads/main/Machine_learning/10_classification/banking.csv', delimiter = ',')

    ### Уберем колонку duration

    df = df.drop('duration', axis=1)
    ### Посмотрим на категориальные колонки

    categorical_columns = df.loc[:,df.dtypes=='object'].columns

    df = df.drop(['loan', 'housing', 'marital'], axis=1)
    categorical_columns = categorical_columns.drop(['loan', 'housing', 'marital'])

    for col in categorical_columns:
    
        ### К колонкам с маленькой размерностью применим one-hot
        if df[col].nunique() < 5:
            one_hot = pd.get_dummies(df[col], prefix=col, drop_first=True).astype(int)
            df = pd.concat((df.drop(col, axis=1), one_hot), axis=1)
            
        ### К остальным - счетчики
        else:
            mean_target = df.groupby(col)['y'].mean()
            df[col] = df[col].map(mean_target)
    
    categorical_columns = df.loc[:,df.dtypes=='object'].columns
    
    df = df.reset_index(drop=True)  
    ordinal = OrdinalEncoder()
    ordinal.fit(df[categorical_columns])
    Ordinal_encoded = ordinal.transform(df[categorical_columns])
    df_ordinal = pd.DataFrame(Ordinal_encoded, columns=categorical_columns)
    df[categorical_columns] = df_ordinal[categorical_columns]
    df.to_csv('df_clear.csv')
    return True

dag_cars = DAG(
    dag_id="train_pipe",
    start_date=datetime(2025, 2, 3),
    concurrency=4,
    schedule_interval=timedelta(minutes=5),
#    schedule="@hourly",
    max_active_runs=1,
    catchup=False,
)
download_task = PythonOperator(python_callable=download_data, task_id = "download_cars", dag = dag_cars)
clear_task = PythonOperator(python_callable=clear_data, task_id = "clear_cars", dag = dag_cars)
train_task = PythonOperator(python_callable=train, task_id = "train_cars", dag = dag_cars)
download_task >> clear_task >> train_task
