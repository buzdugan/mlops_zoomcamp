#!/usr/bin/env python
# coding: utf-8

from pathlib import Path

import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.datasets import *
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold

import mlflow
from mlflow.tracking import MlflowClient
# from prefect import flow, task
# from prefect_aws import S3Bucket

TARGET = 'claim_status'


#@task(name="mlflow_initialization")
def init_mlflow(mlflow_tracking_uri, mlflow_experiment_name):
    client = MlflowClient(mlflow_tracking_uri)

    mlflow.set_tracking_uri(mlflow_tracking_uri)

    try:
        experiment_id = mlflow.create_experiment(mlflow_experiment_name)
    except:
        experiment_id = mlflow.get_experiment_by_name(
            mlflow_experiment_name
        ).experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)

    return client


#@task(name="read_data", retries=3, retry_delay_seconds=2)
def read_dataframe(file_path):
    df = pd.read_csv(file_path)
    
    # Get categ and numeric columns
    categ_cols = [c for c in df.columns if df[c].dtype == 'object']
    num_cols = [c for c in df.columns if c not in categ_cols]
    num_cols.remove(TARGET)

    for column in categ_cols:
        df[column] = df[column].astype('category')
    categ_cols.remove('policy_id')

    # Convert 'is_' columns to integer values
    is_cols = [c for c in df.columns if c.startswith('is_')]
    for column in is_cols:
        df[column] = df[column].map(dict(Yes=1, No=0))
        df[column] = df[column].astype('int16')
        num_cols.append(column)
        categ_cols.remove(column)

    # TODO: Keep all the columns for modelling
    # Remove "is_" columns for faster training
    for column in is_cols:
        num_cols.remove(column)

    df = df[num_cols + categ_cols + [TARGET]]

    return df


#@task(name="split_data")
def create_train_test_datasets(df):
    X, y = df.drop(TARGET, axis=1), df[[TARGET]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


#@task(name="hyperparameter_tuning", log_prints=True)
def hyperparameter_tuning(X_train, y_train, eval_set, eval_metrics):
    # Stratified cross-validation object
    stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Randomized search for hyperparameter tuning
    parameter_gridSearch = RandomizedSearchCV(
        estimator=xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric=eval_metrics,
        early_stopping_rounds=15,
        enable_categorical=True,
        ),

        param_distributions={
        'n_estimators': stats.randint(50, 500),
        'learning_rate': stats.uniform(0.01, 0.75),
        'subsample': stats.uniform(0.25, 0.75),
        'max_depth': stats.randint(1, 8),
        'colsample_bytree': stats.uniform(0.1, 0.75),
        'min_child_weight': [1, 3, 5, 7, 9],
        },

        cv=stratified_cv,
        n_iter=5,
        verbose=False,
        scoring='average_precision',  # aucpr
    )
    
    parameter_gridSearch.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    
    return parameter_gridSearch.best_params_


#@task(name="train_model", log_prints=True)
def train_model(X_train, y_train, X_test, y_test, artifact_path):
    with mlflow.start_run() as run:
        mlflow.set_tag("model", "xgboost")

        # Build the evaluation set & metric list
        eval_set = [(X_train, y_train)]
        eval_metrics = ['aucpr']

        best_params = hyperparameter_tuning(X_train, y_train, eval_set, eval_metrics)
        mlflow.log_params(best_params)

        # Fit model with the best parameters
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric=eval_metrics,
            early_stopping_rounds=15,
            enable_categorical=True,
            **best_params
            )
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

        # Model evaluation
        train_class_preds = model.predict(X_train)
        test_class_preds = model.predict(X_test)
        train_prob_preds = model.predict_proba(X_train)[:, 1]
        test_prob_preds = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "train_roc": roc_auc_score(y_train, train_prob_preds),
            "test_roc": roc_auc_score(y_test, test_prob_preds),
            "train_aucpr": average_precision_score(y_train, train_prob_preds),
            "test_aucpr": average_precision_score(y_test, test_prob_preds),
            "train_accuracy": accuracy_score(y_train, train_class_preds),
            "test_accuracy": accuracy_score(y_test, test_class_preds),
        }
        print("Metrics:", metrics)
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log the model
        mlflow.xgboost.log_model(model, artifact_path=artifact_path)

         # Save reference data for model monitoring
        print("Saving reference data for model monitoring...")
        val_data = pd.concat([X_test, y_test], axis=1)
        val_data['predicted_claim_status'] = test_class_preds
        val_data.to_parquet("data/reference.parquet")

        return run.info.run_id


#@task(name="register_model", log_prints=True)
def register_model(run_id, model_name, artifact_path):    
    mlflow.register_model(
        model_uri=f"runs:/{run_id}/{artifact_path}",
        name=model_name
    )


#@task(name="productionize_model", log_prints=True)
def stage_model(client, run_id, model_name):
    # Get all registered models for model name
    reg_models = client.search_registered_models(
        filter_string=f"name='{model_name}'"
    )

    # Get trained model version and production model run id
    prod_model_run_id = None
    for reg_model in reg_models:
        for model_version in reg_model.latest_versions:
            if model_version.run_id == run_id:
                trained_model_version = model_version.version

            if model_version.current_stage == 'Production':
                prod_model_run_id = model_version.run_id

    # If no model in production, promote the trained model to production
    if not prod_model_run_id:
        client.transition_model_version_stage(
                name=model_name,
                version=trained_model_version,
                stage="Production",
                archive_existing_versions=True,
            )
        print(f'Productionized version {trained_model_version} of {model_name} model.')
    else:
        # Get the metrics for production and trained models
        prod_model_run = client.get_run(prod_model_run_id)
        prod_model_aucpr = prod_model_run.data.metrics['test_aucpr']
        trained_model_run = client.get_run(run_id)
        trained_model_aucpr = trained_model_run.data.metrics['test_aucpr']

        # If trained model's aucpr score better than production model, promote to production else archive
        if trained_model_aucpr > prod_model_aucpr:
            client.transition_model_version_stage(
                name=model_name,
                version=trained_model_version,
                stage="Production",
                archive_existing_versions=True,
            )
            print(f'Productionized version {trained_model_version} of {model_name} model.')
        else:
            client.transition_model_version_stage(
                name=model_name,
                version=trained_model_version,
                stage="Archived",
            )
            print(f'Archived version {trained_model_version} of {model_name} model.')
    

#@flow(name="claim_status_classification_flow_local")
def main_flow():

    mlflow_tracking_uri = "http://127.0.0.1:5000" # run locally
    experiment_name = "claims_status"
    model_name = f"{experiment_name}_classifier"
    artifact_path = "models_mlflow"
    file_path = Path("data/insurance_claims_data.csv")

    print("Connecting to mlflow tracking server...")
    client = init_mlflow(mlflow_tracking_uri, experiment_name)
    print("Connected to mlflow tracking server...")

    df = read_dataframe(file_path)
    X_train, X_test, y_train, y_test = create_train_test_datasets(df)

    print("Model training starting...")
    run_id = train_model(X_train, y_train, X_test, y_test, artifact_path)
    
    print(f"Registering model {model_name} with run_id: {run_id}.")
    register_model(run_id, model_name, artifact_path)
    print(f"Registered model {model_name} with run_id: {run_id}.")

    stage_model(client, run_id, model_name)
    print(f"Staged model {model_name} with run_id: {run_id}.")
    

if __name__ == "__main__":
    main_flow()






