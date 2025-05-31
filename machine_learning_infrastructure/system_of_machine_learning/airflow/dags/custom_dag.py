from airflow.decorators import dag, task
from datetime import datetime, timedelta
from knn_model import train_and_export_knn

@dag(
    schedule=None,
    start_date=datetime.now() - timedelta(days=1),
    catchup=False,
    tags=['ml', 'knn'],
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    description="Train a KNN model on Ghidra features"
)
def train_knn_model():

    @task
    def train_model():
        train_and_export_knn()

    train_model()

dag = train_knn_model()

