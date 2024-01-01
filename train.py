import random
from dvclive import Live
import yaml
import mlflow

with Live(save_dvc_exp=True) as live:
    with mlflow.start_run():
        epochs = yaml.safe_load(open('params.yaml'))['train']['epochs']

        live.log_param("epochs", epochs)
        mlflow.log_param("epochs",epochs)

        for epoch in range(epochs):
            train_acc = epoch + random.random()
            train_loss = epochs - epoch - random.random()
            val_acc = epoch + random.random()
            val_loss = epochs - epoch - random.random()

            live.log_metric("train/accuracy", train_acc)
            live.log_metric("train/loss", train_loss)
            live.log_metric("val/accuracy", val_acc )
            live.log_metric("val/loss", val_loss)

            mlflow.live.log_metric("train/accuracy", train_acc)
            mlflow.live.log_metric("train/loss", train_loss)
            mlflow.live.log_metric("val/accuracy",val_acc)
            mlflow.live.log_metric("val/loss", val_loss)

            live.next_step()