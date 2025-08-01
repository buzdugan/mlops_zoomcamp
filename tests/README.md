The `test` folder contains unit tests and integration tests.
Run the tests from the terminal with `pytest`.

The integration tests can be run in a docker container. First we need to build the docker image with
```bash
docker build -f tests/Dockerfile -t integration-test:v1 .
```

Then run the docker image to create a scored file for yesterday's date on the S3 bucket.
```bash
docker run -it \
	-v ~/.aws:/root/.aws \
	integration-test:v1 
```

