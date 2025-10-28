import pandas as pd
import numpy as np
import mlflow
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def get_latest_model_version(model_name):
    """Получение последней версии модели"""
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    return max([v.version for v in versions]) if versions else 0

def calculate_model_drift(current_metrics, previous_metrics, threshold=0.05):
    """Расчет дрейфа модели"""
    drift_detected = {}
    
    for metric in current_metrics.keys():
        if metric in previous_metrics:
            change = abs(current_metrics[metric] - previous_metrics[metric])
            drift_detected[metric] = {
                'current': current_metrics[metric],
                'previous': previous_metrics[metric], 
                'change': change,
                'drift': change > threshold
            }
    
    return drift_detected

def prepare_retraining_data(events, items, test_size=0.2):
    """Подготовка данных для дообучения с учетом временных разделений"""
    events_sorted = events.sort_values('timestamp')
    
    # Используем последние данные для тестирования
    split_point = events_sorted['timestamp'].quantile(1 - test_size)
    train_data = events_sorted[events_sorted['timestamp'] < split_point]
    test_data = events_sorted[events_sorted['timestamp'] >= split_point]
    
    return train_data, test_data

def log_retraining_metadata(run_id, data_stats, training_params):
    """Логирование метаданных дообучения"""
    with mlflow.start_run(run_id=run_id):
        mlflow.log_params({
            'retraining_timestamp': datetime.now().isoformat(),
            'training_data_size': data_stats.get('training_size', 0),
            'num_features': data_stats.get('num_features', 0),
            'retraining_type': 'scheduled'
        })
        mlflow.log_params(training_params)