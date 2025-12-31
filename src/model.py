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


#Model request packages
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLflowProductionTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config 
  
    def building_model(self):
        logger.info("Building Model")
        model = XGBClassifier()
        logger.info("Building Model - Success")
        return model
    
    def load_split_data(self): 
        logger.info("Reading data")
        X = pd.read_parquet("data/processed/X.parquet")
        y = pd.read_parquet("data/processed/y.parquet")
        logger.info("Reading data - Success")
        X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.3)
        return X_train, X_test,y_train, y_test
    
    def training_model(self): 
        X_train, X_test,y_train, y_test = self.load_split_data()
        logger.info("Training Model")
        model = self.building_model()
        model.fit(X_train, y_train)
        logger.info("Training Model - Success")
        return model
    
    def evaluating_model(self): 
        model = self.training_model()
        X_train, X_test,y_train, y_test = self.load_split_data()
        y_pred = model.predict(X_test)
        model_acc = accuracy_score(y_test, y_pred)
        return model_acc
      
    def run_mlops_pipeline(self):
        """Complete MLOps pipeline"""
        logger.info("Starting MLOps pipeline...")
        try:
            exp_name = self.config['mlflow']['experiment_name']
        except KeyError as e:
            print(f"Missing config key: {e}")
            exp_name = 'Sensor_Prediction'
        mlflow.set_experiment(exp_name)
        with mlflow.start_run(run_name=f"mlops-{datetime.now().strftime('%Y%m%d-%H%M%S')}") as run:
            model = self.training_model()
            model_info = mlflow.xgboost.log_model(
                        xgb_model=model,
                        name="xgboost_model",
            )
            accuracy = self.evaluating_model()
            mlflow.log_metric("accuracy", accuracy)


if __name__ == "__main__": 
    # Configuration of the class 
    CONFIG = {
        
    "environment": "production",
    
    "mlflow": {
        "tracking_uri": os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000'),  
        "experiment_name": "Predictive_maintenance_IOT_sensors"
    }  
    
    }
    
    mlflow_instance = MLflowProductionTrainer(config=CONFIG)
    mlflow_instance.run_mlops_pipeline()
    print(mlflow.__version__)
    