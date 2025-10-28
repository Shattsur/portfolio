# app/model_loader.py

import lightgbm as lgb
import pandas as pd
import numpy as np
import os
import tempfile
import boto3
import logging
import mlflow
from botocore.exceptions import ClientError
import re
from datetime import datetime
from sklearn.metrics import ndcg_score
import json
from prometheus_client import Gauge, Counter

# Метрики для модели 
MODEL_LOAD_STATUS = Gauge(
    'model_load_status',
    'Model load status (1=loaded, 0=error)'
)

MODEL_FEATURES_COUNT = Gauge(
    'model_features_count',
    'Number of features in loaded model'
)

MODEL_LOAD_ERROR_COUNTER = Counter(
    'model_load_errors_total',
    'Total model load errors'
)

logger = logging.getLogger(__name__)

_model = None
_feature_columns = None

# ===========================
# Загрузка модели и фичей из S3
# ===========================
def load_model_from_s3():
    """Загрузка оригинальной модели v2 из S3"""
    logger.info("📥 Загрузка ОРИГИНАЛЬНОЙ модели v2 из S3...")
    
    endpoint_url = os.getenv("MLFLOW_S3_ENDPOINT_URL", "https://storage.yandexcloud.net").strip()
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", "").strip().strip("\"'")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip().strip("\"'")
    bucket = os.getenv("S3_BUCKET_NAME")

    if not all([aws_access_key_id, aws_secret_access_key, bucket]):
        raise ValueError("Отсутствуют обязательные переменные окружения AWS/S3")

    s3 = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    # 🎯 ПУТЬ К НОВОЙ ОРИГИНАЛЬНОЙ МОДЕЛИ v2
    model_key = "mlflow-artifacts/2/2f85e04263a94acbafc5aa5ed76f053c/artifacts/model_binary/model.bin"
    
    # СНАЧАЛА ПРОВЕРИМ СУЩЕСТВОВАНИЕ ФАЙЛА
    try:
        logger.info(f"🔍 Проверка существования модели: {bucket}/{model_key}")
        s3.head_object(Bucket=bucket, Key=model_key)
        logger.info("✅ Модель найдена в S3")
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            logger.error(f"❌ Модель НЕ НАЙДЕНА по пути: {bucket}/{model_key}")
            # Покажем какие файлы есть в S3
            try:
                logger.info("🔍 Поиск доступных моделей в S3...")
                response = s3.list_objects_v2(Bucket=bucket, Prefix="mlflow-artifacts/", Delimiter="/")
                if 'CommonPrefixes' in response:
                    prefixes = [p['Prefix'] for p in response['CommonPrefixes']]
                    logger.info(f"📁 Доступные эксперименты: {prefixes}")
            except Exception as list_error:
                logger.error(f"❌ Ошибка при поиске моделей: {list_error}")
        raise e
    
    logger.info(f"🔗 Загрузка ОРИГИНАЛЬНОЙ модели v2: {bucket}/{model_key}")
    logger.info(f"🎯 Run ID: 2f85e04263a94acbafc5aa5ed76f053c")

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp_file:
        try:
            s3.download_file(bucket, model_key, tmp_file.name)
            model = lgb.Booster(model_file=tmp_file.name)
            logger.info("✅ ОРИГИНАЛЬНАЯ модель v2 успешно загружена из S3")
            return model
        finally:
            if os.path.exists(tmp_file.name):
                os.unlink(tmp_file.name)

def load_features_from_s3():
    logger.info("📥 Загрузка списка фичей из S3...")
    endpoint_url = os.getenv("MLFLOW_S3_ENDPOINT_URL", "https://storage.yandexcloud.net").strip()
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", "").strip().strip("\"'")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip().strip("\"'")
    bucket = os.getenv("S3_BUCKET_NAME")

    if not all([aws_access_key_id, aws_secret_access_key, bucket]):
        logger.warning("⚠️ Нет S3 credentials — используем резервные фичи")
        return get_default_features()

    s3 = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    features_key = "mlflow-artifacts/2/2f85e04263a94acbafc5aa5ed76f053c/artifacts/model_info/features.txt"
    try:
        response = s3.get_object(Bucket=bucket, Key=features_key)
        content = response['Body'].read().decode("utf-8")
        lines = content.strip().split("\n")
        features = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if ". " in line:
                features.append(line.split(". ", 1)[1])
            else:
                features.append(line)
        if features and features[0].lower().startswith("features"):
            features = features[1:]
        logger.info(f"✅ Загрузка фичей из S3: {len(features)} элементов")
        return features
    except Exception as e:
        logger.warning(f"⚠️ Ошибка загрузки features.txt: {e}")
        return get_default_features()

def get_default_features():
    base_features = [
        'user_prev_events',
        'user_prev_unique_items',
        'item_cum_views',
        'item_cum_atc',
        'item_atc_rate_hist',
        'hour',
        'day_of_week',
        'is_weekend'
    ]
    numeric_features = [f'prop_{i}' for i in range(400)]
    return base_features + numeric_features

def get_model():
    """Загрузка ОРИГИНАЛЬНОЙ модели с 421 признаком (новая версия v2)"""
    global _model, _feature_columns
    if _model is None:
        logger.info("🎯 Загрузка ОРИГИНАЛЬНОЙ модели v2 (421 признак)...")
        try:
            _model = load_model_from_s3()
            _feature_columns = load_features_from_s3()
            logger.info(f"✅ ОРИГИНАЛЬНАЯ модель v2 загружена. Фичи: {len(_feature_columns)}")
        except Exception as e:
            logger.error(f"❌ КРИТИЧЕСКАЯ ОШИБКА загрузки модели: {e}")
            raise 
    return _model, _feature_columns


def model_predict(model, X):
    return model.predict(X)

# ===========================
# Вспомогательные функции для подготовки данных
# ===========================
def extract_first_number(x):
    """Извлекает первое число из строки (поддерживает float, int, отрицательные)."""
    if pd.isna(x):
        return np.nan
    match = re.search(r"[-+]?\d*\.?\d+", str(x))
    return float(match.group()) if match else np.nan

def prepare_properties_data(properties_df):
    """Подготовка свойств товаров в числовой формат"""
    logger.info("🔄 Подготовка данных свойств товаров...")
    
    if properties_df is None or properties_df.empty:
        logger.warning("⚠️ Данные свойств товаров отсутствуют или пусты")
        return None
        
    try:
        # Проверяем наличие необходимых колонок
        required_columns = ['itemid', 'property', 'property_value']
        missing_columns = [col for col in required_columns if col not in properties_df.columns]
        
        if missing_columns:
            logger.error(f"❌ Отсутствуют колонки: {missing_columns}")
            logger.info(f"📊 Доступные колонки: {list(properties_df.columns)}")
            return None
        
        # Преобразуем в широкий формат (pivot)
        properties_df = properties_df.copy()
        properties_df['property_value'] = properties_df['property_value'].astype(str)
        
        # Создаем широкий формат
        props_wide = properties_df.pivot_table(
            index='itemid',
            columns='property',
            values='property_value',
            aggfunc='first'
        ).reset_index()
        
        # Извлекаем числа из строковых значений
        props_wide_numeric = props_wide.copy()
        for col in props_wide_numeric.columns:
            if col != 'itemid' and props_wide_numeric[col].dtype == 'object':
                props_wide_numeric[col] = props_wide_numeric[col].apply(extract_first_number)
        
        # Заполняем пропуски и преобразуем типы
        props_wide_numeric = props_wide_numeric.fillna(0)
        
        # Преобразуем все колонки кроме itemid в float32
        for col in props_wide_numeric.columns:
            if col != 'itemid':
                props_wide_numeric[col] = props_wide_numeric[col].astype(np.float32)
        
        logger.info(f"✅ Преобразовано {props_wide_numeric.shape[1] - 1} числовых признаков из свойств.")
        return props_wide_numeric
        
    except Exception as e:
        logger.error(f"❌ Ошибка при подготовке свойств товаров: {e}")
        import traceback
        logger.error(f"📋 Детали ошибки: {traceback.format_exc()}")
        return None

def prepare_ranker_features_for_retraining(
    events_df,
    props_wide_numeric=None,
    target_event='addtocart',
    neg_ratio=3
):
    """
    Подготовка данных для дообучения с ВСЕМИ признаками оригинальной модели
    """
    logger.info("🔄 Подготовка данных для дообучения с сохранением структуры признаков...")
    
    # Конвертация timestamp
    events_df = events_df.copy()
    events_df['timestamp'] = pd.to_datetime(events_df['timestamp'], unit='ms')
    events_df = events_df.sort_values('timestamp')
    
    # 1. Положительные примеры
    positives = events_df[events_df['event'] == target_event][['visitorid', 'itemid', 'timestamp']].copy()
    positives['target'] = 1

    # 2. Негативы: просмотры без addtocart
    views = events_df[events_df['event'] == 'view'][['visitorid', 'itemid', 'timestamp']].copy()
    
    if views.empty:
        true_negatives = pd.DataFrame(columns=['visitorid', 'itemid', 'timestamp', 'target'])
    else:
        # Упрощенная логика для негативов
        views_with_atc = pd.merge(
            views,
            positives[['visitorid', 'itemid']].drop_duplicates(),
            on=['visitorid', 'itemid'],
            how='left',
            indicator=True
        )
        true_negatives = views_with_atc[views_with_atc['_merge'] == 'left_only'][['visitorid', 'itemid', 'timestamp']]
        true_negatives['target'] = 0

    # 3. Объединение и балансировка
    samples = pd.concat([positives, true_negatives], ignore_index=True)
    if samples.empty:
        logger.warning("⚠️ Нет данных для дообучения")
        return pd.DataFrame(), []

    # Балансировка
    pos_count = len(positives)
    neg_count = len(true_negatives)
    
    if neg_count > pos_count * neg_ratio:
        true_negatives_sampled = true_negatives.sample(n=min(neg_count, pos_count * neg_ratio), random_state=42)
        samples_balanced = pd.concat([positives, true_negatives_sampled], ignore_index=True)
    else:
        samples_balanced = samples

    logger.info(f"📊 Балансировка: {len(positives)} позитивов, {len(true_negatives_sampled)} негативов")

    # 4. БАЗОВЫЕ ФИЧИ (те, что реально можем вычислить)
    base_features = []
    
    # User history features
    user_stats = events_df.groupby('visitorid').agg({
        'timestamp': 'count',
        'itemid': 'nunique'
    }).reset_index()
    user_stats.columns = ['visitorid', 'user_prev_events', 'user_prev_unique_items']
    base_features.extend(['user_prev_events', 'user_prev_unique_items'])
    
    samples_balanced = samples_balanced.merge(user_stats, on='visitorid', how='left')

    # Item history features
    item_views = events_df[events_df['event'] == 'view'].groupby('itemid').size().reset_index(name='item_cum_views')
    item_atc = events_df[events_df['event'] == 'addtocart'].groupby('itemid').size().reset_index(name='item_cum_atc')
    
    base_features.extend(['item_cum_views', 'item_cum_atc'])
    
    samples_balanced = samples_balanced.merge(item_views, on='itemid', how='left')
    samples_balanced = samples_balanced.merge(item_atc, on='itemid', how='left')
    
    samples_balanced['item_atc_rate_hist'] = samples_balanced['item_cum_atc'] / (samples_balanced['item_cum_views'] + 1)
    base_features.append('item_atc_rate_hist')

    # 5. Временные фичи
    samples_balanced['hour'] = samples_balanced['timestamp'].dt.hour
    samples_balanced['day_of_week'] = samples_balanced['timestamp'].dt.dayofweek
    samples_balanced['is_weekend'] = (samples_balanced['day_of_week'] >= 5).astype(int)
    
    base_features.extend(['hour', 'day_of_week', 'is_weekend'])

    # 6. Свойства товаров (если доступны)
    if props_wide_numeric is not None:
        samples_balanced = samples_balanced.merge(props_wide_numeric, on='itemid', how='left')
        prop_features = [col for col in props_wide_numeric.columns if col != 'itemid']
        base_features.extend(prop_features)
        logger.info(f"✅ Добавлено {len(prop_features)} признаков из свойств товаров")

    # Заполняем пропуски в вычисленных признаках
    samples_balanced = samples_balanced.fillna(0)
    
    # 7. 🔥 КЛЮЧЕВОЙ ЭТАП: ДОБАВЛЯЕМ ВСЕ ПРИЗНАКИ ИЗ ОРИГИНАЛЬНОЙ МОДЕЛИ
    logger.info("🔥 Добавление всех признаков оригинальной модели...")
    
    # 🔧 ИСПРАВЛЕНИЕ: Используем глобальные feature_columns вместо вызова get_model()
    global _feature_columns
    if _feature_columns is None:
        _, _feature_columns = get_model()
    original_features = _feature_columns
    
    logger.info(f"📊 Оригинальная модель имеет {len(original_features)} признаков")
    
    # СОЗДАЕМ ФИНАЛЬНЫЙ DATAFRAME С ПРАВИЛЬНОЙ СТРУКТУРОЙ
    final_data = pd.DataFrame(index=samples_balanced.index)
    
    # Добавляем служебные колонки
    service_cols = ['visitorid', 'itemid', 'timestamp', 'target']
    for col in service_cols:
        if col in samples_balanced.columns:
            final_data[col] = samples_balanced[col]
    
    # Добавляем ВСЕ признаки из оригинальной модели в правильном порядке
    features_added = 0
    features_missing = 0
    
    for feature in original_features:
        if feature in samples_balanced.columns:
            # Если признак есть в наших данных - используем его
            final_data[feature] = samples_balanced[feature]
            features_added += 1
        elif feature not in service_cols:
            # Если признака нет в данных - заполняем нулями
            final_data[feature] = 0
            features_missing += 1
    
    logger.info(f"✅ Структура данных: {len(final_data)} строк × {len(final_data.columns)} колонок")
    logger.info(f"📊 Признаки: {features_added} вычислено, {features_missing} заполнено нулями")
    logger.info(f"🎯 Используется {len(original_features)} признаков как в оригинальной модели")
    
    return final_data, original_features

def evaluate_model(model, test_data, feature_columns, k_list=[5, 10, 20]):
    """Оценка модели"""
    if test_data.empty:
        return {}
    
    test_data = test_data.copy()
    X_test = test_data[feature_columns]
    test_data['prediction'] = model.predict(X_test)
    
    def calculate_ranking_metrics(df, k_list=k_list):
        metrics = {}
        
        for k in k_list:
            precisions, recalls, ndcgs = [], [], []
            
            for visitor_id, group in df.groupby('visitorid'):
                if len(group) < 2:
                    continue
                    
                # Сортируем по предсказаниям
                top_k = group.nlargest(k, 'prediction')
                
                # Precision@K
                precision = (top_k['target'] == 1).sum() / len(top_k)
                precisions.append(precision)
                
                # Recall@K
                total_relevant = (group['target'] == 1).sum()
                if total_relevant > 0:
                    recall = (top_k['target'] == 1).sum() / total_relevant
                    recalls.append(recall)
                
                # NDCG@K
                try:
                    y_true = group['target'].values.reshape(1, -1)
                    y_score = group['prediction'].values.reshape(1, -1)
                    ndcg = ndcg_score(y_true, y_score, k=k)
                    ndcgs.append(ndcg)
                except:
                    ndcgs.append(0.0)
            
            # ЗАМЕНИЛИ @ на _at_
            metrics[f'ndcg_at_{k}'] = np.mean(ndcgs) if ndcgs else 0
            metrics[f'precision_at_{k}'] = np.mean(precisions) if precisions else 0
            metrics[f'recall_at_{k}'] = np.mean(recalls) if recalls else 0
        
        return metrics
    
    return calculate_ranking_metrics(test_data, k_list)

def save_model_to_s3(model, feature_columns):
    """Сохранение модели и РЕАЛЬНЫХ фичей в S3"""
    logger.info("💾 Сохранение модели в S3...")
    
    endpoint_url = os.getenv("MLFLOW_S3_ENDPOINT_URL", "https://storage.yandexcloud.net").strip()
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", "").strip().strip("\"'")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip().strip("\"'")
    bucket = os.getenv("S3_BUCKET_NAME")

    s3 = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    # Создаем временную директорию
    with tempfile.TemporaryDirectory() as temp_dir:
        # Сохраняем модель
        model_path = os.path.join(temp_dir, "model_retrained.bin")
        model.save_model(model_path)
        
        # Сохраняем РЕАЛЬНЫЕ фичи
        features_path = os.path.join(temp_dir, "features_retrained.txt")
        with open(features_path, 'w') as f:
            f.write(f"Actual features used in retrained model ({len(feature_columns)}):\n")
            for i, feature in enumerate(feature_columns):
                f.write(f"{i+1}. {feature}\n")
        
        # Загружаем в S3
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_key = f"retrained_models/{timestamp}/model.bin"
        features_key = f"retrained_models/{timestamp}/features.txt"
        
        s3.upload_file(model_path, bucket, model_key)
        s3.upload_file(features_path, bucket, features_key)
        
        logger.info(f"✅ Модель сохранена в S3: {bucket}/{model_key}")
        logger.info(f"✅ Реальные фичи сохранены в S3: {bucket}/{features_key}")
        logger.info(f"📊 Сохранено {len(feature_columns)} признаков")

# ===========================
# Основная функция дообучения
# ===========================
def retrain_model_with_mlflow():
    """
    Полное дообучение модели с правильной структурой признаков
    """
    # Используем IP адрес вместо hostname
    # mlflow.set_tracking_uri("http://172.18.0.4:5000")
    mlflow.set_tracking_uri("http://mlflow:5000")
    
    # Загружаем текущую модель
    model, original_feature_columns = get_model()
    
    try:
        # Загружаем данные для дообучения
        events_path = "/app/data/events.parquet"
        
        logger.info("📥 Загрузка данных для дообучения...")
        events = pd.read_parquet(events_path)
        
        logger.info(f"📊 Данные: {len(events)} событий")
        
        # ВРЕМЕННО ОТКЛЮЧАЕМ СВОЙСТВА ТОВАРОВ
        props_wide_numeric = None
        logger.info("⚠️ Свойства товаров временно отключены для тестирования")
        
        # Подготавливаем данные с правильной структурой признаков
        training_data, feature_columns_to_use = prepare_ranker_features_for_retraining(
            events_df=events,
            props_wide_numeric=props_wide_numeric,
            neg_ratio=3
        )
        
        if training_data is None or len(training_data) == 0:
            logger.warning("⚠️ Недостаточно данных для дообучения")
            return None, None, {"error": "Недостаточно данных"}
        
        logger.info(f"📊 Подготовлено данных: {len(training_data)} строк, {len(feature_columns_to_use)} признаков")
        
        # Убеждаемся, что порядок признаков соответствует оригинальной модели
        logger.info(f"🔍 Проверка соответствия признаков: {len(feature_columns_to_use)} vs {len(original_feature_columns)}")
        
        # Разделяем на train/validation
        np.random.seed(42)
        mask = np.random.rand(len(training_data)) < 0.2
        validation_data = training_data[mask]
        train_data = training_data[~mask]
        
        if len(train_data) == 0:
            logger.warning("⚠️ Недостаточно данных после разделения")
            train_data = training_data
            validation_data = training_data
        
        # Группы для ранжирования
        train_groups = train_data.groupby('visitorid').size().values
        val_groups = validation_data.groupby('visitorid').size().values
        
        # Используем ВСЕ признаки в правильном порядке
        train_features = train_data[feature_columns_to_use]
        train_target = train_data['target']
        val_features = validation_data[feature_columns_to_use]
        val_target = validation_data['target']
        
        # 🔥 ИСПОЛЬЗУЕМ СУЩЕСТВУЮЩИЙ ЭКСПЕРИМЕНТ recsys_weekly_retraining
        experiment_name = "recsys_weekly_retraining"
        try:
            # Пробуем найти существующий эксперимент
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                # Создаем новый эксперимент
                experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"📊 Создан новый эксперимент: {experiment_name} (ID: {experiment_id})")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"📊 Используем существующий эксперимент: {experiment_name} (ID: {experiment_id})")
        except Exception as e:
            logger.warning(f"⚠️ Ошибка работы с экспериментом: {e}")
            # Используем default эксперимент как fallback
            experiment_id = "0"
            logger.info(f"📊 Используем default эксперимент: {experiment_id}")
        
        # Начинаем MLflow run в указанном эксперименте
        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=f"retraining_{datetime.now().strftime('%Y%m%d_%H%M')}"
        ) as run:
            
            logger.info(f"🎯 Run создан в эксперименте {experiment_id}: {run.info.run_id}")
            
            # Параметры дообучения
            params = {
                "learning_rate": 0.01,
                "num_boost_round": 50,
                "retraining_date": datetime.now().isoformat(),
                "data_size": len(training_data),
                "num_features": len(feature_columns_to_use),
                "training_samples": len(train_data),
                "validation_samples": len(validation_data),
                "properties_disabled": True,
                "features_preserved": "all_421",  # Все признаки сохранены
                "experiment_name": experiment_name
            }
            
            # Логируем параметры
            mlflow.log_params(params)
            
            # Создаем Dataset
            lgb_train = lgb.Dataset(train_features, label=train_target, group=train_groups)
            lgb_val = lgb.Dataset(val_features, label=val_target, group=val_groups, reference=lgb_train)
            
            # Параметры модели для дообучения (БЕЗ отключения проверки!)
            model_params = {
                'objective': 'lambdarank',
                'metric': 'ndcg',
                'eval_at': [5, 10, 20],
                'learning_rate': params['learning_rate'],
                'verbosity': 1
                # НЕ добавляем predict_disable_shape_check - теперь проверка пройдет!
            }
            
            # 🔥 ПРАВИЛЬНОЕ ДООБУЧЕНИЕ С СУЩЕСТВУЮЩЕЙ МОДЕЛЬЮ
            logger.info("🔧 Дообучение существующей модели...")
            updated_model = lgb.train(
                model_params,
                lgb_train,
                num_boost_round=params['num_boost_round'],
                valid_sets=[lgb_val],
                valid_names=['validation'],
                init_model=model,  # ✅ ИСПОЛЬЗУЕМ СУЩЕСТВУЮЩУЮ МОДЕЛЬ
                keep_training_booster=True,
                callbacks=[lgb.log_evaluation(10)]
            )
            
            # Оценка модели
            metrics = evaluate_model(updated_model, validation_data, feature_columns_to_use)
            
            # Логируем метрики
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Логируем теги
            mlflow.set_tags({
                "model_type": "lightgbm_ranker",
                "purpose": "recommendations",
                "retraining": "true",
                "environment": "production",
                "properties_enabled": "false",
                "features_used": f"all_{len(feature_columns_to_use)}",
                "original_run_id": "a05ede4ec398403f87ac372ae5ea1254",
                "experiment": experiment_name
            })
            
            # Сохраняем модель в MLflow
            mlflow.lightgbm.log_model(
                updated_model,
                "model",
                registered_model_name="lgbm_ranker_optuna"
            )
            
            # Сохраняем модель в S3 с реальными фичами
            save_model_to_s3(updated_model, feature_columns_to_use)
            
            logger.info(f"✅ Дообучение завершено! Run ID: {run.info.run_id}")
            logger.info(f"📊 Метрики: {metrics}")
            logger.info(f"📊 Использовано признаков: {len(feature_columns_to_use)}")
            logger.info(f"📁 Эксперимент: {experiment_name} (ID: {experiment_id})")
            
            return updated_model, feature_columns_to_use, metrics
            
    except Exception as e:
        logger.error(f"❌ Ошибка дообучения: {e}")
        import traceback
        logger.error(f"📋 Детали ошибки: {traceback.format_exc()}")
        return None, None, {"error": str(e)}
    
def load_latest_retrained_model():
    """Загрузка последней дообученной модели из S3"""
    logger.info("📥 Загрузка последней дообученной модели из S3...")
    
    endpoint_url = os.getenv("MLFLOW_S3_ENDPOINT_URL", "https://storage.yandexcloud.net").strip()
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", "").strip().strip("\"'")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip().strip("\"'")
    bucket = os.getenv("S3_BUCKET_NAME")

    s3 = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    try:
        # Ищем последнюю дообученную модель
        response = s3.list_objects_v2(
            Bucket=bucket,
            Prefix="retrained_models/",
            Delimiter="/"
        )
        
        if 'CommonPrefixes' not in response:
            raise Exception("Не найдены дообученные модели в S3")
        
        # Сортируем по дате (последние сначала)
        folders = sorted([p['Prefix'] for p in response['CommonPrefixes']], reverse=True)
        
        if not folders:
            raise Exception("Нет доступных дообученных моделей")
        
        # Берем последнюю модель
        latest_folder = folders[0]
        model_key = f"{latest_folder}model.bin"
        features_key = f"{latest_folder}features.txt"
        
        logger.info(f"🔗 Загрузка модели: {bucket}/{model_key}")
        
        # Загружаем модель
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp_file:
            s3.download_file(bucket, model_key, tmp_file.name)
            model = lgb.Booster(model_file=tmp_file.name)
            os.unlink(tmp_file.name)
        
        # Загружаем признаки
        response = s3.get_object(Bucket=bucket, Key=features_key)
        content = response['Body'].read().decode("utf-8")
        lines = content.strip().split("\n")
        
        features = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("Features used") or ":" in line:
                continue
            if ". " in line:
                features.append(line.split(". ", 1)[1])
            else:
                features.append(line)
        
        logger.info(f"✅ Загружена дообученная модель с {len(features)} признаками")
        return model, features
        
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки дообученной модели: {e}")
        raise    

def load_latest_retrained_model():
    """Загрузка последней дообученной модели из S3"""
    logger.info("📥 Загрузка последней дообученной модели из S3...")
    
    endpoint_url = os.getenv("MLFLOW_S3_ENDPOINT_URL", "https://storage.yandexcloud.net").strip()
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", "").strip().strip("\"'")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip().strip("\"'")
    bucket = os.getenv("S3_BUCKET_NAME")

    if not all([aws_access_key_id, aws_secret_access_key, bucket]):
        raise ValueError("Отсутствуют обязательные переменные окружения AWS/S3")

    s3 = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    try:
        # Ищем папки с дообученными моделями
        response = s3.list_objects_v2(
            Bucket=bucket,
            Prefix="retrained_models/",
            Delimiter="/"
        )
        
        logger.info(f"🔍 Поиск дообученных моделей в {bucket}/retrained_models/")
        
        if 'CommonPrefixes' not in response or not response['CommonPrefixes']:
            raise Exception("Не найдены дообученные модели в S3")
        
        # Сортируем папки по дате (последние сначала)
        folders = sorted([p['Prefix'] for p in response['CommonPrefixes']], reverse=True)
        latest_folder = folders[0]
        
        logger.info(f"🎯 Найдена последняя модель в: {latest_folder}")
        
        # Ключи для модели и фичей
        model_key = f"{latest_folder}model.bin"
        features_key = f"{latest_folder}features.txt"
        
        # Загружаем модель
        logger.info(f"🔗 Загрузка модели: {bucket}/{model_key}")
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp_file:
            s3.download_file(bucket, model_key, tmp_file.name)
            model = lgb.Booster(model_file=tmp_file.name)
            # Очищаем временный файл
            os.unlink(tmp_file.name)
        
        # Загружаем признаки
        logger.info(f"🔗 Загрузка признаков: {bucket}/{features_key}")
        response = s3.get_object(Bucket=bucket, Key=features_key)
        content = response['Body'].read().decode("utf-8")
        lines = content.strip().split("\n")
        
        # Парсим признаки
        features = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("Features used") or line.startswith("Actual features"):
                continue
            if ". " in line:
                # Формат: "1. feature_name"
                features.append(line.split(". ", 1)[1])
            else:
                features.append(line)
        
        # Фильтруем пустые строки
        features = [f for f in features if f and f not in ['visitorid', 'itemid', 'timestamp', 'target']]
        
        logger.info(f"✅ Загружена дообученная модель с {len(features)} признаками")
        logger.info(f"📊 Первые 5 признаков: {features[:5]}")
        
        return model, features
        
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки дообученной модели: {e}")
        raise    