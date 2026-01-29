"""
Production MLflow Training =
Features: Experiment tracking, model registry, reproducibility, testing, CI/CD readiness
"""
import os
import json
import pandas as pd
from datetime import datetime
import mlflow
import logging
import warnings
import boto3
from typing import Dict, Any, Optional, List
import tensorflow as tf
import preprocessing
import pickle

#Model request packages
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')
# Configure loggin.  
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ml_ops_pipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config 
    
    def preprocess_data(self): 
        logger.info("Checking if data is preprocessed")
        filepath = 'data/processed'
        if len(os.listdir(filepath)) == 0:
            logger.info("Processing data")
            df = preprocessing.read_data('data/ai4i2020.csv')
            cleaned_df = preprocessing.clean_column_names(df)
            X,y = preprocessing.prepare_features_and_target(cleaned_df)
            #X is a dataframe with all the features for training, we can directly convert it to a parquet
            X.to_parquet("data/processed/X.parquet") 
            #Y is a series with all the outputs, we have to convert it to a DataFrame by treating the Series as a single column in order to convert it to a parquet
            y.to_frame().to_parquet("data/processed/y.parquet")
        else: 
            logger.info("Preprocessed data is available")
            
    def load_split_data(self,stage): 
        self.preprocess_data()
        logger.info("Reading data")
        X = pd.read_parquet("data/processed/X.parquet")
        y = pd.read_parquet("data/processed/y.parquet")
        logger.info("Reading data - Success")
        X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.3)
        if stage == 1: 
            return X_train, y_train
        else: 
            return X_test, y_test
    
    def building_model(self):
        logger.info("Building Model")
        model = XGBClassifier()
        logger.info("Building Model - Success")
        return model
    
    def training_model(self): 
        X_train, y_train = self.load_split_data(stage=1)
        logger.info("Training Model")
        model = self.building_model()
        model.fit(X_train, y_train)
        logger.info("Training Model - Success")
        return model
    
    def evaluating_model(self,model): 
        X_test, y_test = self.load_split_data(stage=2)
        y_pred = model.predict(X_test)
        model_acc = accuracy_score(y_test, y_pred)
        logger.info("Evalution (Step) - Success")
        return model_acc
    
    def save_model(self,model): 
        logger.info("Saving model")
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        logger.info("Save model - Success")
    
    def run_mlops_pipeline(self):
        """Complete MLOps pipeline"""
        logger.info("Starting MLOps pipeline...")
        logger.info("Starting Training")
        model = self.training_model()
        self.save_model(model)
        logger.info("Starting Evaluation")
        accuracy = self.evaluating_model(model)
        return accuracy


if __name__ == "__main__": 
    # Configuration of the class 
    CONFIG = {
        
    "environment": "production",
    
    }
    
    mlflow_instance = ml_ops_pipeline(config=CONFIG)
    accuracy = mlflow_instance.run_mlops_pipeline()
    