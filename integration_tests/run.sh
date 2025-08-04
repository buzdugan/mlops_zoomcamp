#!/usr/bin/env bash

if [ "${LOCAL_IMAGE_NAME}" == "" ]; then 
    LOCAL_TAG=`date +"%Y-%m-%d-%H-%M"`
    export LOCAL_IMAGE_NAME="integration-test:${LOCAL_TAG}"
    echo "LOCAL_IMAGE_NAME is not set, building a new image with tag ${LOCAL_IMAGE_NAME}"
    docker build -f integration_tests/Dockerfile -t ${LOCAL_IMAGE_NAME} .
else
    echo "no need to build image ${LOCAL_IMAGE_NAME}"
fi

sleep 300
docker run -it \
	-v ~/.aws:/root/.aws \
	${LOCAL_IMAGE_NAME}

ERROR_CODE=$?

if [ ${ERROR_CODE} != 0 ]; then
    exit ${ERROR_CODE}
fi