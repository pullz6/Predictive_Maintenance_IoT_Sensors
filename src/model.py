"""
Production MLflow Training =
Features: Experiment tracking, model registry, reproducibility, testing, CI/CD readiness
"""
import os
import json
import mlflow
import logging
import warnings
import boto3
from typing import Dict, Any, Optional, List
import tensorflow as tf

warnings.filterwarnings('ignore')
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLflowProductionTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config 
        #self.set_up_mlflow()
        self.set_up_aws()
    
    def set_up_mlflow(self):
        """Setup MLflow tracking and registry"""
        # MLflow Tracking
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])
        
        # MLflow Client
        self.mlflow_client = mlflow.MlflowClient()
        
        # Enable autologging
        mlflow.sklearn.autolog()
  
        
    # def building_model(): 
        
        
    # def training_model(): 
    
     # def evaluating_model(): 
     
     # def upload_artifact(): 


if __name__ == "__main__": 
    # Configuration of the class 
    CONFIG = {"environment": "production",
    
    #"mlflow": {
        #"tracking_uri": os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000'),  
       # "experiment_name": "Predictive_maintenance_IOT_sensors"
    #}  
    
    "aws": {
         "access_key_id": os.environ.get('AWS_ACCESS_KEY_ID',''),
        "secret_access_key": os.environ.get('AWS_SECRET_ACCESS_KEY',''),
         "region": os.environ.get('AWS_DEFAULT_REGION','')
    }
    }
    
    mlflow_instance = MLflowProductionTrainer(config=CONFIG)