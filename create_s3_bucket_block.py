from time import sleep
from prefect_aws import S3Bucket, AwsCredentials


def create_aws_creds_block():
    aws_creds_block_obj = AwsCredentials(
        profile_name="mlops-user"
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
    create_aws_creds_block()
    sleep(5)
    create_s3_bucket_block()
