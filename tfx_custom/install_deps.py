pip install apache-airflow[tfx]

export AIRFLOW_HOME=~/airflow
airflow db init

// Start airflow services
airflow scheduler &
airflow webserver &

