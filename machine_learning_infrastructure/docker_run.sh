#!/bin/bash

# Set your username or adjust the paths
USER_NAME=$(whoami)
LOCAL_AIRFLOW_DIR="/home/${USER_NAME}/airflow"
LOCAL_DATA_DIR="/home/${USER_NAME}/data"

# Ensure local directories exist
mkdir -p "$LOCAL_AIRFLOW_DIR" "$LOCAL_DATA_DIR"

# Run Docker container
docker run -it --rm \
    --name airflow_training_pipeline \
    -v "${LOCAL_AIRFLOW_DIR}:/home/airflow" \
    -v "${LOCAL_DATA_DIR}:/home/data" \
    -e AIRFLOW_HOME="/home/airflow" \
    training_pipeline:dev /bin/bash

