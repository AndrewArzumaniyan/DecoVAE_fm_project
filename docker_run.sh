#!/bin/bash
source docker_params

docker run \
    -d \
    --shm-size=20g \
    --memory=200g \
    --cpus=16 \
    --gpus all \
    --network host \
    -e UV_INDEX_STRATEGY=unsafe-best-match \
    -e MLFLOW_SERVER_ALLOWED_HOSTS='*' \
    --user ${DOCKER_USER_ID}:${DOCKER_GROUP_ID} \
    --name ${CONTAINER_NAME} \
    --rm -it --init \
    --mount type=bind,src=/home/${USER}/${SRC},dst=/app \
    -w /app \
    ${CONTAINER_NAME} bash
