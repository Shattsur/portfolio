# app/utils.py

import pandas as pd
import numpy as np
from datetime import datetime

def prepare_features_batch(requests):
    """Пакетная подготовка фичей для нескольких запросов"""
    # Здесь можно добавить логику преобразования фичей
    # которые требуются для вашей модели
    return requests

def validate_features(features_dict, expected_features):
    """Валидация полученных фичей"""
    missing_features = set(expected_features) - set(features_dict.keys())
    if missing_features:
        print(f"⚠️ Отсутствуют фичи: {missing_features}")
    
    return {k: features_dict.get(k, 0) for k in expected_features}