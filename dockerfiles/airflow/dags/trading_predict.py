# dags/flujo_completo_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

# Importamos las tareas desde plugins/tasks/
from tasks.s3_utils import ejemplo_conexion_s3, descargar_dataset
from tasks.procesamiento_utils import leer_y_loguear_minio, split_dataset_minio
from tasks.entrenamiento_utils import (
    simple_mlflow_run,
    train_lightgbm_optuna_minio,
    train_randomforest_optuna_minio,
    train_logisticregression_optuna_minio,
    train_knn_optuna_minio
)
from tasks.prediccion_utils import seleccionar_mejor_modelo, predict_datos_actuales, test_endpoints_predict
