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
        #self.set_up_mlflow()
    
    def set_up_mlflow(self):
        """Setup MLflow tracking and registry"""
        logger.info("Setting up Mlflow")
        # MLflow Tracking
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])
        
        # MLflow Client
        self.mlflow_client = mlflow.MlflowClient()
        
        # Enable autologging
        mlflow.sklearn.autolog()
  
        
    def building_model(self):
        logger.info("Building Model")
        model = XGBClassifier()
        logger.info("Building Model - Success")
        return model
    
        
    def training_model(self): 
        logger.info("Reading data")
        X = pd.read_parquet("data/processed/X.parquet")
        y = pd.read_parquet("data/processed/y.parquet")
        logger.info("Reading data - Success")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        logger.info("Training Model")
        model = self.building_model()
        model.fit(X_train, y_train)
        logger.info("Training Model - Success")
    
     # def evaluating_model(): 
     
     # def upload_artifact():
      
    def run_mlops_pipeline(self):
        """Complete MLOps pipeline"""
        logger.info("Starting MLOps pipeline...")
        
        with mlflow.start_run(run_name=f"mlops-{datetime.now().strftime('%Y%m%d-%H%M%S')}") as run:
            self.building_model()
            self.training_model()


if __name__ == "__main__": 
    # Configuration of the class 
    CONFIG = {"environment": "production",
    
    "mlflow": {
        "tracking_uri": os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000'),  
        "experiment_name": "Predictive_maintenance_IOT_sensors"
    }  
    
    }
    
    mlflow_instance = MLflowProductionTrainer(config=CONFIG)
    mlflow_instance.run_mlops_pipeline()
    print(mlflow.__version__)
    