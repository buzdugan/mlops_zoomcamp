import boto3
from time import sleep
from prefect_aws import S3Bucket, AwsCredentials


def create_aws_creds_block(creds):
    aws_creds_block_obj = AwsCredentials(
        aws_access_key_id=creds.access_key,
        aws_secret_access_key=creds.secret_key,
        aws_session_token=creds.token
    )
    aws_creds_block_obj.save("mlops-aws-creds", overwrite=True)


def create_s3_bucket_block():
    aws_creds = AwsCredentials.load("mlops-aws-creds")
    my_s3_bucket_obj = S3Bucket(
        bucket_name="mlflow-artifacts-remote-claims", 
        credentials=aws_creds
    )
    my_s3_bucket_obj.save(name="mlops-s3-bucket", overwrite=True)


if __name__ == "__main__":

    session = boto3.Session(profile_name="mlops-user" )
    creds = session.get_credentials().get_frozen_credentials()

    create_aws_creds_block(creds)
    sleep(5)
    create_s3_bucket_block()
