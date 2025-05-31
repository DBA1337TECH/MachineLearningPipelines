# Dockerfile for Airflow + TensorFlow training pipeline
FROM ubuntu:oracular AS training_pipeline

# Install system dependencies
RUN apt-get -y update && \
    apt-get install -y python3 python3-pip python3-venv curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create virtual environment
ENV VIRTUAL_ENV=/opt/venv_airflow
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install Python packages
RUN pip install --upgrade pip
RUN pip install apache-airflow==3.0.1 tensorflow-cpu

# Setup Airflow home
ENV AIRFLOW_HOME=/home/airflow
RUN mkdir -p $AIRFLOW_HOME
WORKDIR $AIRFLOW_HOME

