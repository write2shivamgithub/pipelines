import random
import sys
import yaml
from dvclive import Live

with Live(save_dvc_exp=True) as live:
    train_params = yaml.safe_load(open('params.yaml'))['train']
    epochs = train_params['epochs']
    live.log_param("epochs", epochs)
    for epoch in range(epochs):
        live.log_metric("train/accuracy", epoch + random.random())
        live.log_metric("train/loss", epochs - epoch - random.random())
        live.log_metric("val/accuracy",epoch + random.random() )
        live.log_metric("val/loss", epochs - epoch - random.random())
        live.next_step()