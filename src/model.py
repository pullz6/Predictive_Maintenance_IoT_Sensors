import mlflow

with mlflow.start_run():
    mlflow.log_param("lr", 0.001)

    mlflow.log_metric("val_loss", val_loss)