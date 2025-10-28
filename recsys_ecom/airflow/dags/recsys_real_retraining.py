# airflow/dags/recsys_real_retraining.py

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import logging
import requests

logger = logging.getLogger(__name__)

default_args = {
    'owner': 'recsys',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(minutes=10)
}

def trigger_real_retraining(ti):
    """Запуск дообучения с интеллектуальной проверкой статуса"""
    import time
    
    logger.info("🚀 Запуск реального дообучения модели...")
    
    try:
        api_url = "http://recsys_api:8000"
        
        # 1. Сохраняем текущее состояние модели ДО дообучения
        health_before = requests.get(f"{api_url}/health", timeout=10)
        if health_before.status_code == 200:
            model_before = health_before.json()
            last_trained_before = model_before.get('last_trained')
            logger.info(f"📊 Состояние ДО дообучения: обучена {last_trained_before}")
        else:
            logger.warning("⚠️ Не удалось получить состояние модели до дообучения")
            last_trained_before = None
        
        # 2. Запускаем дообучение
        response = requests.post(
            f"{api_url}/retrain",
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ Дообучение запущено: {result}")
            
            # 3. Интеллектуальное ожидание с проверками
            max_wait_time = 300  # 5 минут максимум
            check_interval = 10  # Проверяем каждые 10 секунд
            elapsed_time = 0
            
            logger.info("⏳ Ожидаем завершения дообучения с проверками...")
            
            while elapsed_time < max_wait_time:
                time.sleep(check_interval)
                elapsed_time += check_interval
                
                # Проверяем статус API
                try:
                    health_response = requests.get(f"{api_url}/health", timeout=5)
                    if health_response.status_code == 200:
                        current_status = health_response.json()
                        
                        # Проверяем, обновилась ли модель
                        current_trained = current_status.get('last_trained')
                        if current_trained and current_trained != last_trained_before:
                            logger.info(f"✅ Модель обновлена! Новое время: {current_trained}")
                            return {
                                'status': 'completed',
                                'message': 'Дообучение успешно завершено',
                                'timestamp': datetime.now().isoformat(),
                                'model_updated': True,
                                'wait_time': elapsed_time
                            }
                        
                        logger.info(f"⏰ Ожидание... {elapsed_time}/{max_wait_time}с, модель еще не обновлена")
                    else:
                        logger.warning(f"⚠️ Ошибка проверки здоровья: {health_response.status_code}")
                        
                except requests.exceptions.RequestException as e:
                    logger.warning(f"⚠️ Ошибка подключения к API: {e}")
                
                # Дополнительная проверка: если прошло больше 2 минут, предполагаем проблемы
                if elapsed_time > 120:
                    logger.warning("🕐 Долгое выполнение дообучения (>2 минут), возможно проблемы...")
            
            # Если вышли по таймауту
            logger.warning("⏰ Достигнут максимальный timeout ожидания")
            return {
                'status': 'timeout',
                'message': 'Дообучение не завершилось в ожидаемое время',
                'timestamp': datetime.now().isoformat(),
                'model_updated': False,
                'wait_time': elapsed_time
            }
            
        else:
            logger.error(f"❌ Ошибка запуска дообучения: {response.status_code} - {response.text}")
            return {'status': 'failed', 'error': f"HTTP {response.status_code}"}
            
    except Exception as e:
        logger.error(f"❌ Ошибка при вызове API дообучения: {e}")
        return {'status': 'failed', 'error': str(e)}
    
def check_mlflow_status():
    """Проверка MLflow с использованием IP адреса"""
    logger.info("🔍 Проверка MLflow через IP...")
    
    try:
        # Используем IP вместо hostname для обхода DNS rebinding
        # mlflow_url = "http://172.18.0.4:5000"  # IP из docker сети
        mlflow_url = 'http://mlflow:5000'  # Альтернативный вариант, если DNS работает
        
        # Проверяем доступность
        health_response = requests.get(f"{mlflow_url}/health", timeout=10)
        if health_response.status_code == 200:
            logger.info("✅ MLflow доступен через IP")
        else:
            logger.warning(f"⚠️ MLflow health check: {health_response.status_code}")
            return {'status': 'mlflow_unavailable'}
        
        # Запрос runs
        runs_response = requests.post(
            f"{mlflow_url}/api/2.0/mlflow/runs/search",
            json={
                "experiment_ids": ["0"],
                "max_results": 5,
                "order_by": ["attributes.start_time DESC"]
            },
            timeout=10
        )
        
        if runs_response.status_code == 200:
            runs_data = runs_response.json()
            runs = runs_data.get('runs', [])
            logger.info(f"📈 Найдено runs: {len(runs)}")
            
            if runs:
                latest_run = runs[0]
                run_info = latest_run['info']
                
                logger.info(f"🆕 Последний run: {run_info['run_name']}")
                logger.info(f"🆔 Run ID: {run_info['run_id']}")
                
                return {
                    'status': 'success',
                    'run_id': run_info['run_id'],
                    'run_name': run_info['run_name'],
                    'runs_count': len(runs)
                }
            else:
                logger.info("ℹ️ Runs не найдено")
                return {'status': 'no_runs_found'}
        else:
            logger.warning(f"⚠️ Ошибка поиска runs: {runs_response.status_code}")
            return {'status': 'runs_search_failed'}
            
    except Exception as e:
        logger.error(f"❌ Ошибка проверки MLflow: {e}")
        return {'status': 'error', 'error': str(e)}
    
def _try_alternative_mlflow_check(mlflow_url):
    """Альтернативный способ проверки MLflow"""
    try:
        logger.info("🔄 Пробуем альтернативный способ проверки...")
        
        # Простой запрос к основному endpoint
        response = requests.get(mlflow_url, timeout=10)
        if response.status_code in [200, 304]:
            logger.info("✅ MLflow UI доступен")
            
            # Пробуем получить runs без фильтрации
            runs_response = requests.post(
                f"{mlflow_url}/api/2.0/mlflow/runs/search",
                json={
                    "max_results": 3
                },
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if runs_response.status_code == 200:
                runs_data = runs_response.json()
                runs = runs_data.get('runs', [])
                if runs:
                    return {
                        'status': 'success_alternative',
                        'run_id': runs[0]['info']['run_id'],
                        'runs_count': len(runs)
                    }
            
            return {'status': 'mlflow_available_no_runs'}
        else:
            return {'status': 'mlflow_unavailable'}
            
    except Exception as e:
        logger.warning(f"⚠️ Альтернативная проверка не удалась: {e}")
        return {'status': 'alternative_check_failed'}

with DAG(
    'recsys_real_retraining',
    default_args=default_args,
    description='Реальное дообучение модели с сохранением в MLflow и S3',
    schedule_interval=timedelta(days=7),
    catchup=False,
    max_active_runs=1,
    tags=['recsys', 'retraining', 'mlops', 'production']
) as dag:

    start = BashOperator(
        task_id='start',
        bash_command='echo "🚀 Starting Real RecSys Model Retraining"'
    )

    check_mlflow = PythonOperator(
        task_id='check_mlflow_status',
        python_callable=check_mlflow_status
    )

    trigger_retraining = PythonOperator(
        task_id='trigger_real_retraining',
        python_callable=trigger_real_retraining,
        execution_timeout=timedelta(minutes=10)  # Увеличиваем таймаут для задачи
    )

    end = BashOperator(
        task_id='end',
        bash_command='echo "✅ Real RecSys Model Retraining Pipeline Completed"'
    )

    start >> check_mlflow >> trigger_retraining >> end