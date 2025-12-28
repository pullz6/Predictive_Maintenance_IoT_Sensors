import pandas as pd
import numpy as np 
import logging 
from sklearn.preprocessing import LabelEncoder

# Basic setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_data(path): 
    logger.info("Loading Data")
    df = pd.read_csv(path)
    return df

def fill_nulls(df): 
    df = df.fillna(df.mean(numeric_only=True))
    logger.info("Filling the nulls")
    return df

def clean_column_names(df):
    """Remove invalid characters from column names"""
    logger.info("Cleaning the column names")
    df_clean = df.copy()
    df_clean.columns = [col.replace('[', '').replace(']', '').replace('<', '')
                       for col in df_clean.columns]
    return df_clean

def prepare_features_and_target(df):
    """
    Features: Sensor data and operational parameters
    Target: Machine failure (binary)
    """
    logger.info("Preprocessing the dataframe")
    # FEATURES - What causes failures
    feature_columns = [
        'Air temperature K', 'Process temperature K',
        'Rotational speed rpm', 'Torque Nm',
        'Tool wear min', 'Type'
    ]

    X = df[feature_columns].copy()

    # Feature engineering
    X['temp_difference'] = X['Process temperature K'] - X['Air temperature K']
    X['power_estimate'] = (X['Torque Nm'] * X['Rotational speed rpm']) / 9549
    X['wear_rate'] = X['Tool wear min'] / (X['Rotational speed rpm'] + 1)

    # Encode categorical
    le = LabelEncoder()
    X['Type_encoded'] = le.fit_transform(X['Type'])

    # Drop original categorical column
    X = X.drop(['Type'], axis=1)

    # TARGET - What we want to predict
    y = df['Machine failure']
    
    logger.info("Exporting X, y into parquets")
    return X, y
        
if __name__ == "__main__":
    df = read_data('data/ai4i2020.csv')
    cleaned_df = clean_column_names(df)
    X,y = prepare_features_and_target(cleaned_df)
    #X is a dataframe with all the features for training, we can directly convert it to a parquet
    X.to_parquet("data/processed/X.parquet") 
    #Y is a series with all the outputs, we have to convert it to a DataFrame by treating the Series as a single column in order to convert it to a parquet
    y.to_frame().to_parquet("data/processed/y.parquet")