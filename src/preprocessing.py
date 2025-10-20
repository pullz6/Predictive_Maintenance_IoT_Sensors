import pandas as pd
import numpy as np 

def read_data(path): 
    df = pd.read_csv(path)
    return df 

df = read_data('data/ai4i2020.csv')
print(df.info())

#fix nulls