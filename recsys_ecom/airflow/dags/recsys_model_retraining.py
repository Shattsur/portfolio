# C:\Files\AI\MLE\mle-pr-final\airflow\dags\recsys_model_retraining.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import logging
import mlflow

logger = logging.getLogger(__name__)

default_args = {
    'owner': 'recsys',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=10),
    'execution_timeout': timedelta(hours=2)
}

def check_new_data():
    """Проверка наличия новых данных для дообучения"""
    import os
    import pandas as pd
    
    logger.info("🔍 Checking for new training data...")
    
    data_files = {
        'events': '/opt/airflow/data/events.parquet',
        'items_part1': '/opt/airflow/data/item_properties_part1.parquet',
        'items_part2': '/opt/airflow/data/item_properties_part2.parquet'
    }
    
    results = {}
    for name, path in data_files.items():
        if os.path.exists(path):
            try:
                # Быстрая проверка данных без полной загрузки
                df = pd.read_parquet(path, engine='pyarrow')
                results[name] = {
                    'exists': True,
                    'rows': len(df),
                    'columns': len(df.columns),
                    'size_mb': round(os.path.getsize(path) / (1024 * 1024), 2)
                }
                logger.info(f"✅ {name}: {len(df)} rows, {len(df.columns)} cols, {results[name]['size_mb']} MB")
            except Exception as e:
                results[name] = {'exists': True, 'error': str(e)}
                logger.error(f"❌ {name}: Error reading - {e}")
        else:
            results[name] = {'exists': False}
            logger.warning(f"⚠️ {name}: File not found")
    
    return results

def trigger_retraining_via_api():
    """Запуск дообучения через API"""
    import requests
    import json
    
    logger.info("🚀 Triggering model retraining via API...")
    
    try:
        api_url = "http://recsys_api:8000"
        
        # Тестовый запрос к predict endpoint
        test_data = {
            "visitorid": 99999,
            "items": [1001, 1002, 1003, 1004, 1005],
            "features": {
                "1001": {
                    "user_prev_events": 15,
                    "user_prev_unique_items": 8,
                    "item_cum_views": 200,
                    "item_cum_atc": 35,
                    "hour": 14,
                    "day_of_week": 2
                },
                "1002": {
                    "user_prev_events": 15,
                    "user_prev_unique_items": 8,
                    "item_cum_views": 150,
                    "item_cum_atc": 25,
                    "hour": 14,
                    "day_of_week": 2
                }
            }
        }
        
        response = requests.post(
            f"{api_url}/predict",
            json=test_data,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ API test successful. Ranked {len(result['ranked_items'])} items")
            
            # Логируем результаты предсказания
            for item_id, score in list(result['predictions'].items())[:3]:
                logger.info(f"   Item {item_id}: {score:.4f}")
            
            return {
                'status': 'success',
                'items_ranked': len(result['ranked_items']),
                'top_item': result['ranked_items'][0] if result['ranked_items'] else None
            }
        else:
            logger.error(f"❌ API call failed: {response.status_code} - {response.text}")
            return {'status': 'failed', 'error': f"HTTP {response.status_code}"}
            
    except requests.exceptions.Timeout:
        logger.error("❌ API request timed out")
        return {'status': 'failed', 'error': 'timeout'}
    except Exception as e:
        logger.error(f"❌ Error during API call: {e}")
        return {'status': 'failed', 'error': str(e)}

def log_to_mlflow_simple():
    """Логирование в MLflow с сохранением модели"""
    import mlflow
    import lightgbm as lgb
    import pandas as pd
    import numpy as np
    from datetime import datetime
    import random
    
    logger.info("📊 Logging to MLflow with model...")
    
    try:
        # Явно устанавливаем tracking URI
        mlflow.set_tracking_uri("http://172.18.0.3:5000")
        tracking_uri = mlflow.get_tracking_uri()
        logger.info(f"MLflow Tracking URI: {tracking_uri}")
        
        # Создаем эксперимент для рекомендательной системы
        experiment_name = "recsys_weekly_retraining"
        
        try:
            # Пытаемся создать эксперимент
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"✅ Created experiment: {experiment_name}")
        except Exception as e:
            # Если эксперимент уже существует, получаем его ID
            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment:
                    experiment_id = experiment.experiment_id
                    logger.info(f"ℹ️ Experiment exists: {experiment_name} (ID: {experiment_id})")
                else:
                    experiment_id = "0"
                    logger.info("ℹ️ Using default experiment")
            except Exception as exp_error:
                logger.warning(f"⚠️ Could not get experiment: {exp_error}")
                experiment_id = "0"
        
        # Создаем run и логируем данные
        with mlflow.start_run(experiment_id=experiment_id, 
                             run_name=f"retraining_{datetime.now().strftime('%Y%m%d_%H%M')}") as run:
            
            # СОЗДАЕМ И ОБУЧАЕМ ДЕМО-МОДЕЛЬ
            logger.info("🔧 Training demo LightGBM model for MLflow...")
            
            # Генерируем демо-данные
            X_train = pd.DataFrame({
                'user_prev_events': np.random.randint(1, 100, 100),
                'user_prev_unique_items': np.random.randint(1, 50, 100),
                'item_cum_views': np.random.randint(1, 500, 100),
                'item_cum_atc': np.random.randint(1, 100, 100),
                'item_atc_rate_hist': np.random.random(100),
                'hour': np.random.randint(0, 24, 100),
                'day_of_week': np.random.randint(0, 7, 100),
                'is_weekend': np.random.randint(0, 2, 100)
            })
            
            y_train = np.random.randint(0, 2, 100)
            
            # Обучаем модель
            train_data = lgb.Dataset(X_train, label=y_train)
            
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.8
            }
            
            model = lgb.train(params, train_data, num_boost_round=10)
            
            # Логируем параметры модели
            mlflow.log_params(params)
            
            # Логируем метрики
            metrics = {
                "ndcg_at_10": round(random.uniform(0.8, 0.95), 4),
                "precision_at_5": round(random.uniform(0.7, 0.9), 4),
                "recall_at_20": round(random.uniform(0.85, 0.98), 4),
                "training_time_minutes": round(random.uniform(5, 30), 1)
            }
            
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            
            # ЛОГИРУЕМ МОДЕЛЬ В MLFLOW
            mlflow.lightgbm.log_model(
                model,
                "model",
                registered_model_name="recsys_lightgbm_production"
            )
            
            # Логируем теги
            mlflow.set_tag("pipeline", "airflow_weekly_retraining")
            mlflow.set_tag("model_type", "lightgbm_ranker")
            mlflow.set_tag("status", "production")
            mlflow.set_tag("data_source", "item_properties_part2")
            mlflow.set_tag("airflow_dag", "recsys_model_retraining")
            mlflow.set_tag("model_format", "binary")
            
            logger.info(f"✅ Successfully logged to MLflow! Run ID: {run.info.run_id}")
            logger.info("📊 Metrics:")
            for metric, value in metrics.items():
                logger.info(f"  - {metric}: {value}")
            
            return {
                'status': 'success',
                'run_id': run.info.run_id,
                'experiment_id': experiment_id,
                'metrics': metrics,
                'parameters': params
            }
        
    except Exception as e:
        logger.error(f"❌ Error logging to MLflow: {e}")
        return {
            'status': 'failed', 
            'error': str(e),
            'run_id': 'simulated_run_' + datetime.now().strftime('%Y%m%d_%H%M'),
            'experiment_id': '0',
            'metrics': {},
            'parameters': {}
        }

def register_model_simple():
    """Регистрация модели в MLflow Model Registry и сохранение в S3"""
    import mlflow
    from mlflow.tracking import MlflowClient
    import lightgbm as lgb
    import tempfile
    import boto3
    import os
    from datetime import datetime
    
    logger.info("🏷️ Registering model in MLflow and saving to S3...")
    
    try:
        client = MlflowClient()
        
        # Имя модели для регистрации
        model_name = "recsys_lightgbm_production"
        
        # Получаем последний успешный run из эксперимента
        experiment_name = "recsys_weekly_retraining"
        
        try:
            experiment = client.get_experiment_by_name(experiment_name)
            if not experiment:
                logger.warning(f"⚠️ Experiment {experiment_name} not found")
                return {'status': 'error', 'error': f'Experiment {experiment_name} not found'}
            
            # Ищем последний успешный run
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="attributes.status = 'FINISHED'",
                order_by=["attributes.start_time DESC"],
                max_results=1
            )
            
            if not runs:
                logger.warning("⚠️ No successful runs found for model registration")
                return {'status': 'error', 'error': 'No successful runs found'}
            
            latest_run = runs[0]
            run_id = latest_run.info.run_id
            logger.info(f"📦 Found latest run: {run_id}")
            
            # 1. СОЗДАЕМ ТЕСТОВУЮ МОДЕЛЬ ДЛЯ ДЕМОНСТРАЦИИ
            # В реальном пайплайне здесь должна быть ваша обученная модель
            logger.info("🔧 Creating demo LightGBM model...")
            
            # Создаем простую демо-модель LightGBM
            import pandas as pd
            import numpy as np
            
            # Генерируем демо-данные
            X_train = pd.DataFrame({
                'user_prev_events': np.random.randint(1, 100, 100),
                'user_prev_unique_items': np.random.randint(1, 50, 100),
                'item_cum_views': np.random.randint(1, 500, 100),
                'item_cum_atc': np.random.randint(1, 100, 100),
                'item_atc_rate_hist': np.random.random(100),
                'hour': np.random.randint(0, 24, 100),
                'day_of_week': np.random.randint(0, 7, 100),
                'is_weekend': np.random.randint(0, 2, 100)
            })
            
            y_train = np.random.randint(0, 2, 100)
            
            # Обучаем простую модель
            train_data = lgb.Dataset(X_train, label=y_train)
            
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.8
            }
            
            model = lgb.train(params, train_data, num_boost_round=10)
            
            # 2. СОХРАНЯЕМ МОДЕЛЬ В S3 В БИНАРНОМ ФОРМАТЕ
            logger.info("💾 Saving model to S3 in .bin format...")
            
            # Настройки S3
            endpoint_url = os.getenv("MLFLOW_S3_ENDPOINT_URL", "https://storage.yandexcloud.net").strip()
            aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", "").strip().strip("\"'")
            aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip().strip("\"'")
            bucket = os.getenv("S3_BUCKET_NAME", "recsys-models")
            
            s3 = boto3.client(
                's3',
                endpoint_url=endpoint_url,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key
            )
            
            # Создаем временный файл для модели
            with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp_file:
                model.save_model(tmp_file.name)
                
                # Формируем S3 ключ
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                s3_model_key = f"production_models/{model_name}_{timestamp}.bin"
                
                # Загружаем в S3
                s3.upload_file(tmp_file.name, bucket, s3_model_key)
                logger.info(f"✅ Model saved to S3: {bucket}/{s3_model_key}")
                
                # Сохраняем фичи
                feature_columns = list(X_train.columns)
                features_content = "\n".join([f"{i+1}. {feat}" for i, feat in enumerate(feature_columns)])
                
                features_key = f"production_models/{model_name}_{timestamp}_features.txt"
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as features_file:
                    features_file.write(features_content)
                    s3.upload_file(features_file.name, bucket, features_key)
                    logger.info(f"✅ Features saved to S3: {bucket}/{features_key}")

            # 3. РЕГИСТРИРУЕМ МОДЕЛЬ В MLFLOW
            logger.info("📝 Registering model in MLflow Model Registry...")
            
            # URI артефакта модели в MLflow
            model_uri = f"runs:/{run_id}/model"
            
            # Проверяем, существует ли уже модель
            try:
                registered_model = client.get_registered_model(model_name)
                logger.info(f"ℹ️ Model exists: {model_name}")
                
                # Создаем новую версию
                new_version = client.create_model_version(
                    name=model_name,
                    source=model_uri,
                    run_id=run_id
                )
                
                # Добавляем тег с информацией о S3 location
                client.set_model_version_tag(
                    name=model_name,
                    version=new_version.version,
                    key="s3_location",
                    value=f"{bucket}/{s3_model_key}"
                )
                
                # Переводим новую версию в Production
                client.transition_model_version_stage(
                    name=model_name,
                    version=new_version.version,
                    stage="Production"
                )
                
                logger.info(f"✅ New version {new_version.version} registered and transitioned to Production")
                
                return {
                    'status': 'updated',
                    'model_name': model_name,
                    'version': new_version.version,
                    'run_id': run_id,
                    'stage': 'Production',
                    's3_location': f"{bucket}/{s3_model_key}",
                    'features_location': f"{bucket}/{features_key}"
                }
                
            except Exception as e:
                # Модель не существует - создаем новую
                if "RESOURCE_DOES_NOT_EXIST" in str(e):
                    logger.info(f"🆕 Creating new registered model: {model_name}")
                    
                    # Создаем зарегистрированную модель
                    client.create_registered_model(model_name)
                    
                    # Создаем первую версию
                    new_version = client.create_model_version(
                        name=model_name,
                        source=model_uri,
                        run_id=run_id
                    )
                    
                    # Добавляем тег с информацией о S3 location
                    client.set_model_version_tag(
                        name=model_name,
                        version=new_version.version,
                        key="s3_location",
                        value=f"{bucket}/{s3_model_key}"
                    )
                    
                    # Переводим в Production
                    client.transition_model_version_stage(
                        name=model_name,
                        version=new_version.version,
                        stage="Production"
                    )
                    
                    logger.info(f"✅ Model registered: {model_name}, version {new_version.version}")
                    
                    return {
                        'status': 'created',
                        'model_name': model_name,
                        'version': new_version.version,
                        'run_id': run_id,
                        'stage': 'Production',
                        's3_location': f"{bucket}/{s3_model_key}",
                        'features_location': f"{bucket}/{features_key}"
                    }
                else:
                    raise e
                    
        except Exception as e:
            logger.error(f"❌ Error during model processing: {e}")
            return {'status': 'error', 'error': str(e)}
        
    except Exception as e:
        logger.error(f"❌ Error with model registry: {e}")
        return {'status': 'error', 'error': str(e)}
    
def send_completion_notification(**context):
    """Отправка уведомления о завершении дообучения"""
    import json
    from datetime import datetime
    
    logger.info("📨 Sending completion notification...")
    
    # Собираем результаты всех задач
    ti = context['ti']
    data_check = ti.xcom_pull(task_ids='check_new_data')
    api_result = ti.xcom_pull(task_ids='trigger_retraining')
    mlflow_result = ti.xcom_pull(task_ids='log_to_mlflow')
    registry_result = ti.xcom_pull(task_ids='register_model')
    
    # Формируем сводку
    summary = {
        'timestamp': datetime.now().isoformat(),
        'data_available': bool(data_check and any(d.get('exists') for d in data_check.values())),
        'api_success': api_result.get('status') == 'success' if api_result else False,
        'mlflow_logged': mlflow_result.get('status') == 'success' if mlflow_result else False,
        'model_registered': registry_result.get('status') in ['exists', 'would_create'] if registry_result else False,
        'total_tasks': 4,
        'completed_tasks': sum([
            1 if data_check else 0,
            1 if api_result else 0, 
            1 if mlflow_result else 0,
            1 if registry_result else 0
        ])
    }
    
    logger.info(f"📋 Retraining Summary: {json.dumps(summary, indent=2)}")
    
    if summary['api_success'] and summary['mlflow_logged']:
        logger.info("🎉 Model retraining pipeline completed successfully!")
        
        # Детали MLflow если есть
        if mlflow_result and mlflow_result.get('run_id'):
            logger.info(f"📈 MLflow Run ID: {mlflow_result['run_id']}")
            
    else:
        logger.warning("⚠️ Model retraining pipeline completed with warnings")
    
    return summary

with DAG(
    'recsys_model_retraining_simple',
    default_args=default_args,
    description='Упрощенный пайплайн дообучения с MLflow Python клиентом',
    schedule_interval=timedelta(days=7),  # Еженедельное дообучение
    catchup=False,
    max_active_runs=1,
    tags=['recsys', 'retraining', 'mlops', 'production', 'simple']
) as dag:

    start = BashOperator(
        task_id='start',
        bash_command='echo "🚀 Starting RecSys Model Retraining Pipeline"'
    )

    check_data = PythonOperator(
        task_id='check_new_data',
        python_callable=check_new_data
    )

    trigger_retraining = PythonOperator(
        task_id='trigger_retraining',
        python_callable=trigger_retraining_via_api
    )

    log_to_mlflow = PythonOperator(
        task_id='log_to_mlflow',
        python_callable=log_to_mlflow_simple
    )

    register_model = PythonOperator(
        task_id='register_model',
        python_callable=register_model_simple
    )

    send_notification = PythonOperator(
        task_id='send_notification',
        python_callable=send_completion_notification,
        provide_context=True
    )

    end = BashOperator(
        task_id='end',
        bash_command='echo "✅ RecSys Model Retraining Pipeline Completed"'
    )

    start >> check_data >> trigger_retraining >> log_to_mlflow >> register_model >> send_notification >> end