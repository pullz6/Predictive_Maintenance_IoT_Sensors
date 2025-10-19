import pandas as pd
import numpy as np 
import seaborn as sns 

def read_data(path): 
    df = pd.read_csv(path)
    return df 


df = read_data('data/ai4i2020.csv')
print(df.head())