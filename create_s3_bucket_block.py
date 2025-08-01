from time import sleep
from prefect_aws import S3Bucket, AwsCredentials

import sys
sys.path.append("src")
import utils


def create_aws_creds_block(prefect_block_aws, profile_name):
    aws_creds_block_obj = AwsCredentials(
        profile_name=profile_name
    )
    aws_creds_block_obj.save(prefect_block_aws, overwrite=True)


def create_s3_bucket_block(bucket_name, prefect_block_aws, prefect_block_s3):
    aws_creds = AwsCredentials.load(prefect_block_aws)
    my_s3_bucket_obj = S3Bucket(
        bucket_name=bucket_name, 
        credentials=aws_creds
    )
    my_s3_bucket_obj.save(name=prefect_block_s3, overwrite=True)


if __name__ == "__main__":

    config = utils.load_config(file_path="config.yaml")
    bucket_name = config['bucket_name']
    prefect_block_aws = config['prefect_block_aws']
    prefect_block_s3 = config['prefect_block_s3']
    profile_name = config['profile_name']

    create_aws_creds_block(prefect_block_aws, profile_name)
    sleep(5)
    create_s3_bucket_block(bucket_name, prefect_block_aws, prefect_block_s3)
