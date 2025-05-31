# Start from Ubuntu base
FROM ubuntu:oracular AS training_pipeline

# System prep
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-venv curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Virtual environment setup
ENV VIRTUAL_ENV=/opt/venv_airflow
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install core Python packages
RUN pip install --upgrade pip

RUN apt-get update && apt-get install -y \
    graphviz \
    && apt-get clean

# Install Python binding
RUN pip install graphviz

# Install Airflow
RUN pip install apache-airflow==3.0.1

# Install ML and utility libraries
RUN pip install \
    scikit-learn \
    tensorflow-cpu \
    numpy \
    pandas \
    requests \
    gradio

# Set Airflow home
ENV AIRFLOW_HOME=/home/airflow
RUN mkdir -p $AIRFLOW_HOME
WORKDIR $AIRFLOW_HOME

# Copy DAGs (assumes docker-compose context has airflow/dags/)
COPY airflow/dags $AIRFLOW_HOME/dags

# Expose Airflow port
EXPOSE 8080

