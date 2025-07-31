#!/usr/bin/env python
# coding: utf-8

import yaml

import mlflow
from mlflow.tracking import MlflowClient

def load_config(file_path="config.yaml"):
    with open(file_path) as stream:
        try:
            config = yaml.safe_load(stream)
            print(config)
        except yaml.YAMLError as exc:
            print(exc)

    return config
