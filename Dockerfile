FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ARG DOCKER_NAME=docker
ARG DOCKER_USER_ID=1000
ARG DOCKER_GROUP_ID=1000

# Install uv globally
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
ENV UV_INDEX_STRATEGY=unsafe-best-match

# System tools
RUN apt-get update && apt-get install -y \
    bash \
    git \
    curl \
    nano \
    vim \
    sudo \
    tmux \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /app
WORKDIR /app

# Create user
RUN groupadd -g ${DOCKER_GROUP_ID} ${DOCKER_NAME} && \
    useradd -m \
    -s /bin/bash \
    -u ${DOCKER_USER_ID} \
    -g ${DOCKER_GROUP_ID} \
    ${DOCKER_NAME} && \
    echo "${DOCKER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Give project ownership to user
RUN chown -R ${DOCKER_NAME}:${DOCKER_NAME} /app

USER ${DOCKER_NAME}
RUN echo "source /app/.venv/bin/activate" >> ~/.bashrc