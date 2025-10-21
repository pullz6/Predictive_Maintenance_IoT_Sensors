import pandas as pd
import numpy as np 

def read_data(path): 
    df = pd.read_csv(path)
    return df

def fill_nulls(df): 
    df = df.fillna(df.mean(numeric_only=True))
    return df

def data_check(df): 
    iot_rules = {
    'UDI': {'required': True, 'dtype': 'int64'},
    'Product ID': {'required': True, 'dtype': 'object'},
    'Type': {'required': True, 'dtype': 'object'},
    'Air temperature [K]': {'required': True, 'dtype': 'float64'},
    'Process temperature [K]': {'required': True, 'dtype': 'float64'},
    'Rotational speed [rpm]': {'required': True, 'dtype': 'int64'},
    'Torque [Nm]': {'required': True, 'dtype': 'float64'},
    'Tool wear [min]': {'required': True, 'dtype': 'int64'},
    'Machine failure': {'required': True, 'dtype': 'int64'},
    'TWF': {'required': True, 'dtype': 'int64'},
    'HDF': {'required': True, 'dtype': 'int64'},
    'PWF': {'required': True, 'dtype': 'int64'},
    'OSF': {'required': True, 'dtype': 'int64'},
    'RNF': {'required': True, 'dtype': 'int64'}}
    
    validation_results = {}
    for column, rule in iot_rules.items():
        if column not in df.columns:
            if rule.get('required', False):
                validation_results[column] = f"REQUIRED column missing"
            continue
            
        column_errors = []
        
        # Data type validation
        if 'dtype' in rule:
            if rule['dtype'] == 'numeric' and not np.issubdtype(df[column].dtype, np.number):
                column_errors.append("Should be numeric")
            elif rule['dtype'] == 'categorical' and not df[column].dtype == 'object':
                column_errors.append("Should be categorical")
                
        if column_errors:
            validation_results[column] = column_errors
    
    return validation_results
        
def run_prep ():
    df = read_data('data/ai4i2020.csv')
    validation_results = data_check(df)
    print(df.info())
    print(validation_results)
    return df