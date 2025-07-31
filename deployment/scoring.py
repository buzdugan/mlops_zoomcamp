#!/usr/bin/env python
# coding: utf-8

import os
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from prefect import flow, task
from prefect_aws import S3Bucket

import sys
sys.path.append("src")
import utils


@task(name="read_data", retries=3, retry_delay_seconds=2)
def read_dataframe(file_path, target, quick_train):
    return utils.read_dataframe(file_path, target, quick_train)


@task(name="get_production_model", log_prints=True)
def get_prod_model(client, model_name):
    # Get all registered models for model name
    reg_models = client.search_registered_models(
        filter_string=f"name='{model_name}'"
    )

    # Get production model run id and model id
    prod_model_run_id = None
    prod_model_model_id = None
    for reg_model in reg_models:
        for model_version in reg_model.latest_versions:
            if model_version.current_stage == 'Production':
                prod_model_run_id = model_version.run_id
                prod_model_model_id = model_version.source.replace('models:/', '') 
                break

    if prod_model_run_id:
        print(f"Production model run_id for {model_name}: {prod_model_run_id}")
        return prod_model_run_id, prod_model_model_id
    else:   
        print(f"No production model found for {model_name}.")


@task(name="load_model", log_prints=True)
def load_model(model_id, experiment_id, bucket_name):
    prod_model = f's3://{bucket_name}/{experiment_id}/models/{model_id}/artifacts/'

    print(f"Loading model from {prod_model}...")
    model = mlflow.pyfunc.load_model(prod_model)
    return model


@task(name="apply_model", log_prints=True)
def apply_model(model, run_id, df, output_path):

    df['predicted_claim_status'] = model.predict(df)
    df['model_run_id'] = run_id
    
    print(f"Saving the predictions to {output_path}...")
    df.to_csv(output_path, index=False)
    print(df.head(3))


@flow(name="claim_status_scoring_flow", log_prints=True)
def score_claim_status():

    config = utils.load_config(file_path="config.yaml")
    mlflow_tracking_uri = config['deployment']['cloud']['mlflow_tracking_uri']
    experiment_name = config['experiment_name']
    model_name = config['model_name']
    target = config['target']
    quick_train = config['quick_train']
    bucket_name = config['bucket_name']
    prefect_block_s3 = config['prefect_block_s3']

    print("Connecting to mlflow registry server...")
    client = MlflowClient(mlflow_tracking_uri)
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
    print(f"Experiment ID for {experiment_name}: {experiment_id}")

    os.environ["AWS_PROFILE"] = config['profile_name']  # AWS profile name
    s3_bucket_block = S3Bucket.load(prefect_block_s3)
    s3_bucket_block.download_folder_to_path(from_folder="data", to_folder="data")

    # Define the input and output file paths
    yesterday = datetime.now() - timedelta(1)
    yesterday_str = yesterday.strftime('%Y_%m_%d')
    input_file_path = Path(config['modelling_data_path'])
    scored_data_path_prefix = Path(config['scored_data_path_prefix'])
    output_file_path = Path(f"{scored_data_path_prefix}_{yesterday_str}.csv")

    print(f"Reading data from {input_file_path}...")
    df = read_dataframe(input_file_path, target, quick_train)
    # Sample the data for scoring
    df = df.sample(n=1000, random_state=42).drop(columns=[target])
    
    print(f"Getting production model from registry...")
    run_id, model_id = get_prod_model(client, model_name)
        
    print(f"Loading model with model_id = {model_id}...")
    model = load_model(model_id, experiment_id, bucket_name)

    print(f"Scoring the data using model with run_id = {run_id}...")
    apply_model(model, run_id, df, output_file_path)
    print(f"Scored the data.")


if __name__ == "__main__":
    score_claim_status()
