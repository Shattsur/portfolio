# C:\Files\AI\MLE\mle-pr-final\airflow\dags\api_retraining.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import requests
import logging

logger = logging.getLogger(__name__)

default_args = {
    'owner': 'recsys',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

def call_retraining_api():
    """Вызов API для дообучения модели"""
    try:
        api_url = "http://recsys_api:8000"
        
        # Пример вызова predict endpoint для проверки работы
        test_data = {
            "visitorid": 12345,
            "items": [1001, 1002, 1003],
            "features": {
                "1001": {"user_prev_events": 10, "item_cum_views": 50},
                "1002": {"user_prev_events": 10, "item_cum_views": 30},
                "1003": {"user_prev_events": 10, "item_cum_views": 20}
            }
        }
        
        response = requests.post(
            f"{api_url}/predict",
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ API call successful. Ranked items: {result['ranked_items']}")
            return "API retraining check completed successfully"
        else:
            logger.error(f"❌ API call failed: {response.status_code} - {response.text}")
            return f"API call failed: {response.status_code}"
            
    except Exception as e:
        logger.error(f"❌ Error calling API: {e}")
        return f"Error: {str(e)}"

def health_check():
    """Проверка здоровья всех сервисов"""
    services = {
        'recsys_api': 'http://recsys_api:8000/health',
        'mlflow': 'http://mlflow:5000'
    }
    
    results = {}
    
    for service, url in services.items():
        try:
            response = requests.get(url, timeout=10)
            results[service] = {
                'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                'status_code': response.status_code
            }
            logger.info(f"✅ {service}: {results[service]['status']}")
        except Exception as e:
            results[service] = {
                'status': 'unreachable',
                'error': str(e)
            }
            logger.error(f"❌ {service}: {e}")
    
    return results

with DAG(
    'recsys_api_operations',
    default_args=default_args,
    description='Операции с RecSys API и мониторинг здоровья',
    schedule_interval=timedelta(hours=6),  # Каждые 6 часов
    catchup=False,
    tags=['recsys', 'api', 'monitoring']
) as dag:

    start = BashOperator(
        task_id='start',
        bash_command='echo "🔍 Starting RecSys API monitoring"'
    )

    health_check_task = PythonOperator(
        task_id='health_check',
        python_callable=health_check
    )

    api_test_task = PythonOperator(
        task_id='test_api',
        python_callable=call_retraining_api
    )

    end = BashOperator(
        task_id='end',
        bash_command='echo "✅ RecSys API monitoring completed"'
    )

    start >> health_check_task >> api_test_task >> end