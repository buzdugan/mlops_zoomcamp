import os
import datetime
import time
import random
import logging 
import uuid
import pytz
import pandas as pd
import io
import psycopg
import joblib

from pathlib import Path
from evidently import DataDefinition, Dataset, Report
from evidently.metrics import ValueDrift, DriftedColumnsCount, MissingValueCount

import mlflow
from mlflow.tracking import MlflowClient
from prefect import flow, task
from prefect.deployments import run_deployment
from prefect_aws import S3Bucket

import sys
sys.path.append("src")
import utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")


SEND_TIMEOUT = 10

CREATE_TABLE_STATEMENT = """
drop table if exists drift_metrics;
create table drift_metrics (
	timestamp timestamp,
	prediction_drift float,
	num_drifted_columns integer,
	share_missing_values float
)
"""

CONNECTION_STRING = "host=localhost port=5432 user=postgres password=example"
CONNECTION_STRING_DB = CONNECTION_STRING + " dbname=test"


@task(name="get_num_cat_features", log_prints=True)
def get_num_cat_features(file_path, target, quick_train):
	df = utils.read_dataframe(file_path, target, quick_train)

	# Get categ and numeric columns
	categ_cols = [c for c in df.columns if df[c].dtype == 'category']
	num_cols = [c for c in df.columns if c not in categ_cols]
	num_cols.remove(target)

	return num_cols, categ_cols


@task(name="load_reference_data", log_prints=True)
def load_reference_data(file_path):
	df = pd.read_parquet(file_path)
	return df


@task(name="create_unseen_data", log_prints=True)
def create_unseen_data(file_path, random_state, target, quick_train):
	df = utils.read_dataframe(file_path, target, quick_train)
    # Sample the data for monitoring
	return df.sample(n=1000, random_state=random_state)


@task(name="get_prod_model", log_prints=True)
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


@task(name="apply_model_to_data", log_prints=True)
def apply_model_to_data(model, run_id, df):
	df['predicted_claim_status'] = model.predict(df)
	df['model_run_id'] = run_id
	return df


@task(name="prep_db", log_prints=True)
def prep_db():
	with psycopg.connect("host=localhost port=5432 user=postgres password=example", autocommit=True) as conn:
		res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
		if len(res.fetchall()) == 0:
			conn.execute("create database test;")
		with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example") as conn:
			conn.execute(CREATE_TABLE_STATEMENT)


@task(name="calculate_metrics_postgresql", log_prints=True)
def calculate_metrics_postgresql(curr, i, unseen_df, reference_data, file_path, target, quick_train, prediction):

	begin = datetime.datetime(2025, 7, 1, 0, 0)

	num_features, cat_features = get_num_cat_features(file_path, target, quick_train)
	data_definition = DataDefinition(
		numerical_columns=num_features + [prediction],
		categorical_columns=cat_features,
	)

	report = Report(metrics = [
		ValueDrift(column=prediction),
		DriftedColumnsCount(),
		MissingValueCount(column=prediction),
	])

	print("Importing unseen data...")
	unseen_df = Dataset.from_pandas(unseen_df, data_definition=data_definition)
	print("Importing unseen data...")
	reference_data = Dataset.from_pandas(reference_data, data_definition=data_definition)

	run = report.run(reference_data=reference_data, current_data=unseen_df)
	result = run.dict()

	prediction_drift = result['metrics'][0]['value']
	num_drifted_columns = result['metrics'][1]['value']['count']
	share_missing_values = result['metrics'][2]['value']['share']

	curr.execute(
		"insert into drift_metrics(timestamp, prediction_drift, num_drifted_columns, share_missing_values) values (%s, %s, %s, %s)",
		(begin + datetime.timedelta(i), prediction_drift, num_drifted_columns, share_missing_values)
	)

	return result


@flow(name="batch_monitoring_backfill_flow", log_prints=True)
# async 
def batch_monitoring_backfill():

	config = utils.load_config(file_path="config.yaml")
	mlflow_tracking_uri = config['deployment']['cloud']['mlflow_tracking_uri']
	experiment_name = config['experiment_name']
	model_name = config['model_name']
	target = config['target']
	prediction = config['prediction']
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

	# Identify and load Production model
	print(f"Getting production model from registry...")
	run_id, model_id = get_prod_model(client, model_name)
	print(f"Loading model with model_id = {model_id}...")
	model = load_model(model_id, experiment_id, bucket_name)

	input_file_path = Path(config['modelling_data_path'])
	reference_data = load_reference_data(Path(config['reference_data_path']))

	print("Reference data loaded...")
	print(reference_data.columns)
	reference_data = apply_model_to_data(model, run_id, reference_data.drop(columns=[target, prediction]))
	print("Reference data scored...")

	prep_db()
	last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
	with psycopg.connect(CONNECTION_STRING_DB, autocommit=True) as conn:

		for i in range(0, 5):

			# Generate random unseen data
			print(f"Creating randomly generated unseen data number {i}...")
			unseen_df = create_unseen_data(input_file_path, i, target, quick_train)

			# Score unseen data
			print(f"Scoring the data using model with run_id = {run_id}...")
			unseen_df = apply_model_to_data(model, run_id, unseen_df.drop(columns=[target]))
			print(f"Scored the data.")

			with conn.cursor() as curr:
				result = calculate_metrics_postgresql(
					curr, i, unseen_df, reference_data, input_file_path, target, quick_train, prediction
					)

			# if result['metrics'][0]['value'] >= 0.5:
			# 	print("Drift detected, retraining model...")
				
			# 	try:
			# 		flow_run = await run_deployment(
			# 			name="claim_status_classification_flow_local/claims_status_classification_local",
			# 			timeout=0
			# 		)
			# 		print(f"Successfully triggered retraining: {flow_run.id}")
			# 	except Exception as e:
			# 		print(f"Failed to trigger retraining: {e}")
			# 		raise

			# 	break
			# else:
			# 	print("No drift was detected.")

			new_send = datetime.datetime.now()
			seconds_elapsed = (new_send - last_send).total_seconds()
			if seconds_elapsed < SEND_TIMEOUT:
				time.sleep(SEND_TIMEOUT - seconds_elapsed)
			while last_send < new_send:
				last_send = last_send + datetime.timedelta(seconds=10)
			logging.info("data sent")


if __name__ == '__main__':
	batch_monitoring_backfill()
