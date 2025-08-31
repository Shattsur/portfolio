# dags/air_etl.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import numpy as np
import os

# Пути через переменные окружения (можно задать в docker-compose или в Airflow Variables)
SOURCE_PATH = os.getenv("SOURCE_PATH")
DEST_PATH = os.getenv("DEST_PATH")



def extract(**context):
    """Извлекаем исходные данные"""
    src = os.getenv("SOURCE_PATH")
    if not src:
        raise ValueError("SOURCE_PATH пуст. Проверь .env и блок environment в docker-compose.yaml")

    if not os.path.exists(src):
        raise FileNotFoundError(f"Файл не найден в контейнере: {src}. "
                                f"Проверь volume-монтирование и что файл есть на хосте.")

    # Пытаемся автоматически определить разделитель
    try:
        df = pd.read_csv(src, sep=None, engine="python")
    except Exception:
        # fallback на запятую
        df = pd.read_csv(src)

    context['ti'].xcom_push(key='raw_data', value=df.to_json(orient="records"))
    return f"Extracted {len(df)} rows from {src}"


def transform(**context):
    """Ресемплинг и генерация фич"""
    raw_json = context['ti'].xcom_pull(key='raw_data')
    df = pd.read_json(raw_json)

    # преобразование даты
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')

    # ресемплинг по часу
    df = df.resample('1H').sum()

    # базовые фичи
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek

    # циклические фичи
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # лаги
    for lag in [1, 2, 3, 24, 48, 72, 168]:
        df[f'lag_{lag}'] = df['num_orders'].shift(lag)

    # скользящие статистики
    df['rolling_mean_24'] = df['num_orders'].shift(1).rolling(24).mean()
    df['rolling_std_24'] = df['num_orders'].shift(1).rolling(24).std()

    # удаляем ненужные
    df = df.drop(['hour', 'day_of_week'], axis=1)

    df = df.dropna()

    context['ti'].xcom_push(key='transformed_data', value=df.to_json())
    return f"Transformed dataset shape: {df.shape}"


def load(**context):
    """Сохраняем результат в CSV"""
    dst = os.getenv("DEST_PATH")
    if not dst:
        raise ValueError("DEST_PATH пуст. Проверь .env и docker-compose.yaml")

    folder = os.path.dirname(dst)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    data = context['ti'].xcom_pull(key='transformed_data', task_ids='transform')
    if data is None:
        raise ValueError("Нет данных в XCom от transform")

    df = pd.read_json(data, orient="records")
    df.to_csv(dst, index=False)

    return f"Saved {len(df)} rows to {dst}"


with DAG(
    dag_id="air_etl",
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,  # запускать вручную
    catchup=False,
    tags=["etl", "csv", "taxi"],
) as dag:

    t1 = PythonOperator(
        task_id="extract",
        python_callable=extract
    )

    t2 = PythonOperator(
        task_id="transform",
        python_callable=transform
    )

    t3 = PythonOperator(
        task_id="load",
        python_callable=load
    )

    t1 >> t2 >> t3
