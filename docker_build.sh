#!/bin/bash
source docker_params

docker build -t ${CONTAINER_NAME} . \
    --build-arg DOCKER_NAME=${DOCKER_NAME} \
    --build-arg DOCKER_USER_ID=${DOCKER_USER_ID} \
    --build-arg DOCKER_GROUP_ID=${DOCKER_GROUP_ID}