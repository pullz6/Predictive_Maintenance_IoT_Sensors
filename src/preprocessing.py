import pandas as pd
import numpy as np 

def read_data(path): 
    df = pd.read_csv(path)
    return df

def fill_nulls(df): 
    df = df.fillna(df.mean(numeric_only=True))
    return df

def check_datatypes(): 
    '''Checking if all the data types are matching'''
    check = {'UDI': int, 'Product ID': string , 'Type': string, }
    return 'True'

df = read_data('data/ai4i2020.csv')
print(df.info())

#fix nulls