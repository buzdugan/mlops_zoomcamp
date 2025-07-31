# Insurance Claims Dataset
This repository is the final project for the MLOps Zoomcamp.


# Problem Description
The project demonstrates the data and modelling pipelines built based on the main aspects of MLOps, such as modelling experimentation and tracking, model registry, workflow orchestration, model deployment and monitoring. 


# Problem Statement and Objective
The objective is to simulate a production-ready MLOps pipeline where 
- a model is trained at the start of every month with data up to and including the previous month
- the applications received daily are scored as a batch job every night and used for filtering out risky customers and help manage the reserve that the business needs to hold.


# Technology Used
- python
- MLFlow for model experimentation, tracking, and registry
- Prefect for workflow orchestration
- AWS for cloud infrastructure: EC2 instance, RDS database and S3 bucket
- docker for deployment (daily scoring)


# Steps Overview
The order of the pipeline is as follows:
- Data Collection
- Model Experimentation, Tracking, and Orchestration
- Model Deployment
- Monitoring and Orchestration


## Data Collection
The data used for this project was downloaded from Kaggle [Insurance Claims Dataset](https://www.kaggle.com/datasets/litvinenko630/insurance-claims?resource=download) and stored in the data folder. It has 58,592 rows with information about customers and their claim status. 
The model built for this project asses the risk associated with insuring a particular policyholder based on their characteristics and historical claim behavior, predicting the likelihood of a customer claiming.
A data dictionary can be found [here](https://www.kaggle.com/datasets/litvinenko630/insurance-claims/data).


## Model Experimentation and Tracking
I started experimenting with a local version of MLFlow and Prefect with code from python script model_training/training_local.py, which has the following steps:
- connect to MLFlow tracking server
- read dataframe from csv file
- create train test datasets
- hyperparameter tuning
- train the model with the best hyperparameters 
- register the model to MLFlow model registry
- stage the model to Production if no model exists already or if the AUC PR score is better then the current Production model

Amazon EC2, Amazon RDS and Amazon S3 were setup to host the MLFlow tracking server, to store MLFlow metadata and artifacts respectively.
In the final version of the code, the data used for training is also stored in the same S3 bucket.


## Model Orchestration
The current model orchestration can be done in a local Prefect server or on Prefect Cloud.


# Steps to Replicate the Project - Version 1. Run locally
For the purpose of testing the scoring_local.py script locally with Prefect, you will need to fork this repo and push your own **mlartifacts** folder that gets generated after the model train to the repo. 
This is because Prefect pulls the code from the github repo and needs all the data and artifacts there.

- Setup the Local Environment
- Run the MLFlow Server
- Run the Prefect Server
- Create and Run the Training Deployment
- Create and Run the Scoring Deployment

## Setup the Local Environment
Create virtual environment and install requirements.txt.
Run the below
```bash
pip install pipenv
pipenv shell
pipenv install -r requirements.txt
```

## Run the MLFlow Server
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Go to http://127.0.0.1:5000 to access the server.
If we don't specify a local path to the model artifacts, by default MLFlow will save the experiment and runs in the local folder in 
**mlartifacts/<experiment_id>/<run_id>/artifacts/**.

## Run the Prefect Server
Open a new terminal and activate the virtual environment.
```bash
pipenv shell
prefect server start
```
Go to http://127.0.0.1:4200 to access the server.

## Create and Run the Training Deployment
Open a new terminal and activate the virtual environment. First create a new work pool and then the deployment. 
```bash
pipenv shell

prefect work-pool create mlops_zoomcamp_pool --type process

prefect deploy model_training/training_local.py:main_flow -n claims_status_classification_local -p mlops_zoomcamp_pool
prefect worker start --pool mlops_zoomcamp_pool
```

Open a new terminal and activate the virtual environment.
```bash
pipenv shell

prefect deployment run 'claim_status_classification_flow_local/claims_status_classification_local'
```

<p align="center">
  <img width="90%" src="images/prefect_server_training_local.png" alt="Prefect training_local run">
</p>


You can run the deployment again either from the terminal or from the UI.
This will create a new model that will be compared to the existing one and the worst one will be Archived. 
<p align="center">
  <img width="90%" src="images/mlflow_training_local_registry.png" alt="MLFlow training_local registry 1">
</p>

After you've run the training deployment, push the **mlartifacts** folder that got generated to the repo. 
This is because Prefect pulls the code from the github repo and needs all the data and artifacts there in order to find the model for scoring.


## Create and Run the Scoring Deployment
```bash
prefect deploy deployment/scoring_local.py:score_claim_status -n claims_status_scoring_local -p mlops_zoomcamp_pool
prefect worker start -p mlops_zoomcamp_pool
```

Open new terminal and activate the virtual environment.
```bash
pipenv shell
prefect deployment run 'claim_status_scoring_flow_local/claims_status_scoring_local'
```
<p align="center">
  <img width="90%" src="images/prefect_server_scoring_local.png" alt="Prefect scoring_local run">
</p>
At this point the flow for scoring will run, and it will create a new datasets scored_dataset_<today_date>.csv inside the temporary storage in Prefect. 