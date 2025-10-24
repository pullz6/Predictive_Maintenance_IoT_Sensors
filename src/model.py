"""
Production MLflow Training =
Features: Experiment tracking, model registry, reproducibility, testing, CI/CD readiness
"""

import mlflow
import logging
import warnings
warnings.filterwarnings('ignore')
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLflowProductionTrainer:
    def __init__(self,experiement_name:str, tracking_uri: Optional[str] = None):
        pass
    
def mlflow_model_training (): 
    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

    # Create a new MLflow Experiment
    mlflow.set_experiment("Predictive Maintenance")


