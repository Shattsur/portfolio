# tests/mlflow_test.py

import os
import requests
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# Получаем URI из переменных окружения
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')

print(f"Testing MLflow connection to: {MLFLOW_TRACKING_URI}")

try:
    # Проверяем доступность MLflow
    health_response = requests.get(f"{MLFLOW_TRACKING_URI}/health", timeout=5)
    print(f"✅ MLflow health check: {health_response.status_code}")
    
    # Проверяем API runs
    response = requests.post(
        f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/runs/search",
        json={
            "experiment_ids": ["0"],
            "max_results": 10
        },
        timeout=10
    )
    print(f"✅ MLflow API status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Found {len(data.get('runs', []))} runs")
        print(f"✅ Response: {data}")
    else:
        print(f"❌ Error response: {response.text}")
        
except requests.exceptions.ConnectionError:
    print(f"❌ Cannot connect to MLflow at {MLFLOW_TRACKING_URI}")
except requests.exceptions.Timeout:
    print("❌ MLflow request timeout")
except Exception as e:
    print(f"❌ MLflow error: {e}")