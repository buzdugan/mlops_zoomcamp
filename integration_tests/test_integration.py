import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import boto3
import pytest
from botocore.exceptions import ClientError

sys.path.append("src")
sys.path.append("deployment")
import scoring

import utils


def check_s3_file_exists(bucket_name, object_key):
    s3 = boto3.resource('s3')

    try:
        s3.Object(bucket_name, object_key).load()
    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            return False
        else:
            raise
    else:
        return True


def test_integration():
    config = utils.load_config(file_path="config.yaml")
    bucket_name = config['bucket_name']
    os.environ["AWS_PROFILE"] = config['profile_name']

    # Define the input and output file paths
    yesterday = datetime.now() - timedelta(1)
    yesterday_str = yesterday.strftime('%Y_%m_%d')
    scored_data_path_prefix = config['scored_data_path_prefix']
    output_file_path = f"{scored_data_path_prefix}_{yesterday_str}.csv"

    scoring.score_claim_status()

    scoring_output_exists = check_s3_file_exists(
        bucket_name,
        output_file_path,
    )

    assert scoring_output_exists == True, 'Scoring output file does not exist'

    print('Integration test passed')


if __name__ == '__main__':
    pytest.main([__file__])
