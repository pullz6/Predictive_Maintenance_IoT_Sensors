import pandas as pd
import numpy as np 

def read_data(path): 
    df = pd.read_csv(path)
    return df

def fill_nulls(df): 
    df = df.fillna(df.mean(numeric_only=True))
    return df

def check_ints(): 
    iot_rules = {
    'sensor_id': {'required': True, 'dtype': 'categorical', 'unique': False},
    'timestamp': {'required': True, 'dtype': 'datetime'},
    'temperature': {'required': True, 'dtype': 'numeric', 'min_value': -40, 'max_value': 125},
    'vibration': {'required': False, 'dtype': 'numeric', 'min_value': 0},
    'status': {'required': True, 'allowed_values': ['normal', 'warning', 'critical']}}


df = read_data('data/ai4i2020.csv')
print(df.info())

#fix nulls