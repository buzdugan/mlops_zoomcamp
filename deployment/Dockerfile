# Steps to package codes to docker and run from root directory

FROM python:3.12

RUN pip install -U pip
RUN pip install pipenv 

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["config.yaml", "src/utils.py", "create_s3_bucket_block.py", "deployment/scoring.py", "./"]

ENTRYPOINT ["sh", "-c", "python ./create_s3_bucket_block.py && python ./scoring.py"]
