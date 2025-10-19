import pandas as pd
import numpy as np 
import seaborn as sns 
import preprocessing as prep


df = prep.read_data('data/ai4i2020.csv')
print(df.head())