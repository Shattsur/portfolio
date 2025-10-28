# app/main.py

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import logging
from datetime import datetime
from app.schemas import PredictRequest, PredictResponse, RetrainResponse
from app.model_loader import get_model, model_predict, retrain_model_with_mlflow, load_latest_retrained_model
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST
from starlette.responses import Response
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Сначала создаем app
app = FastAPI(
    title="RecSys Ranker API",
    description="API для ранжирования и дообучения рекомендательной модели",
    version="1.1.0"
)

# Метрики Prometheus
PREDICTION_COUNTER = Counter(
    'model_predictions_total', 
    'Total number of predictions', 
    ['status', 'model_type']
)

PREDICTION_DURATION = Histogram(
    'model_prediction_duration_seconds',
    'Prediction duration in seconds',
    ['model_type']
)

RETRAIN_COUNTER = Counter(
    'model_retrain_total',
    'Total number of retraining attempts',
    ['status']
)

API_REQUEST_COUNTER = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status_code']
)

API_REQUEST_DURATION = Histogram(
    'api_request_duration_seconds',
    'API request duration in seconds'
)

# Middleware для отслеживания запросов
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    API_REQUEST_DURATION.observe(process_time)
    API_REQUEST_COUNTER.labels(
        method=request.method,
        endpoint=request.url.path,
        status_code=response.status_code
    ).inc()
    
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные переменные
model = None
feature_columns = None
last_trained = None

# Endpoint для метрик Prometheus
@app.get("/metrics")
async def metrics():
    from prometheus_client import generate_latest
    data = generate_latest()
    # Убедимся что данные заканчиваются правильно
    if not data.endswith(b'\n# EOF\n'):
        data = data.rstrip() + b'\n# EOF\n'
    return Response(
        data,
        media_type='text/plain; version=0.0.4; charset=utf-8'
    )

@app.on_event("startup")
async def startup_event():
    global model, feature_columns, last_trained
    try:
        model, feature_columns = get_model()
        last_trained = datetime.now().isoformat()
        logger.info(f"✅ ОРИГИНАЛЬНАЯ модель v2 загружена. Фичи: {len(feature_columns)}, Run ID: 2f85e04263a94acbafc5aa5ed76f053c")
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки ОРИГИНАЛЬНОЙ модели v2: {e}")
        raise e

@app.get("/")
async def root():
    return {
        "message": "RecSys Ranker API is running!",
        "docs": "/docs",
        "health": "/health",
        "retrain": "/retrain"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "features_count": len(feature_columns) if feature_columns else 0,
        "last_trained": last_trained
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    start_time = time.time()
    
    if not request.items:
        PREDICTION_COUNTER.labels(status='error', model_type='original_v2').inc()
        raise HTTPException(status_code=400, detail="Список товаров не может быть пустым")
    
    try:
        data = []
        for item_id in request.items:
            row = {"visitorid": request.visitorid, "itemid": item_id}
            row.update(request.features.get(str(item_id), {}))
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Заполняем недостающие колонки нулями
        missing_cols = set(feature_columns) - set(df.columns)
        for col in missing_cols:
            df[col] = 0
        
        df = df[feature_columns].fillna(0)
        predictions = model_predict(model, df)
        pred_dict = dict(zip(request.items, predictions.tolist()))
        ranked_items = [item for item, _ in sorted(pred_dict.items(), key=lambda x: x[1], reverse=True)]

        # Логируем успешное предсказание
        PREDICTION_COUNTER.labels(status='success', model_type='original_v2').inc()
        PREDICTION_DURATION.labels(model_type='original_v2').observe(time.time() - start_time)
        
        logger.info(f"✅ Предсказание для {request.visitorid}: {len(request.items)} товаров")
        return PredictResponse(visitorid=request.visitorid, predictions=pred_dict, ranked_items=ranked_items)

    except Exception as e:
        PREDICTION_COUNTER.labels(status='error', model_type='original_v2').inc()
        logger.error(f"❌ Ошибка предсказания: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/features")
async def get_features():
    return {"features": feature_columns, "count": len(feature_columns)}

@app.post("/retrain", response_model=RetrainResponse)
async def retrain(background_tasks: BackgroundTasks):
    if model is None:
        RETRAIN_COUNTER.labels(status='error').inc()
        raise HTTPException(status_code=500, detail="Модель не загружена")
    
    logger.info("🔄 Запуск дообучения модели...")
    
    try:
        # Запускаем дообучение в фоне
        background_tasks.add_task(_retrain_background)
        RETRAIN_COUNTER.labels(status='started').inc()
        return RetrainResponse(
            status="retraining_started", 
            message="Дообучение запущено в фоне. Модель будет обновлена в MLflow и S3."
        )
    except Exception as e:
        RETRAIN_COUNTER.labels(status='error').inc()
        logger.error(f"❌ Ошибка запуска дообучения: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
def _retrain_background():
    """Фоновая задача дообучения БЕЗ автоматического обновления модели"""
    global last_trained
    try:
        logger.info("⚙️ Дообучение запущено (модель в памяти НЕ обновится автоматически)...")
        new_model, new_features, metrics = retrain_model_with_mlflow()
        
        if new_model is not None:
            last_trained = datetime.now().isoformat()
            RETRAIN_COUNTER.labels(status='completed').inc()
            logger.info(f"✅ Дообучение завершено. Метрики: {metrics}")
            logger.info("💡 Модель сохранена в MLflow/S3. Используйте /switch_model для активации")
        else:
            RETRAIN_COUNTER.labels(status='failed').inc()
            logger.error(f"❌ Дообучение не удалось: {metrics}")
            
    except Exception as e:
        RETRAIN_COUNTER.labels(status='error').inc()
        logger.error(f"💥 Ошибка дообучения: {e}")
        
@app.post("/switch_model")
async def switch_model(model_type: str = "original"):
    """
    Переключение между оригинальной и дообученной моделью
    """
    global model, feature_columns, last_trained
    
    logger.info(f"🔄 Запрос переключения на модель: {model_type}")
    
    if model_type == "retrained":
        try:
            # Пробуем загрузить дообученную модель
            model, feature_columns = load_latest_retrained_model()
            last_trained = datetime.now().isoformat()
            logger.info(f"✅ Переключено на дообученную модель. Признаков: {len(feature_columns)}")
            return {
                "status": "switched", 
                "model_type": "retrained",
                "features_count": len(feature_columns),
                "message": "Загружена последняя дообученная модель",
                "warning": "⚠️ ВНИМАНИЕ: Дообученная модель имеет проблемы с качеством (413 нулевых признаков из 421)"
            }
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки дообученной модели: {e}")
            # Возвращаем оригинальную модель как fallback
            model, feature_columns = get_model()
            return {
                "status": "fallback", 
                "model_type": "original",
                "features_count": len(feature_columns),
                "message": f"Не удалось загрузить дообученную модель: {str(e)}. Используется оригинальная модель"
            }
    
    elif model_type == "original":
        try:
            # Загружаем оригинальную модель
            model, feature_columns = get_model()
            last_trained = datetime.now().isoformat()
            logger.info(f"✅ Переключено на оригинальную модель. Признаков: {len(feature_columns)}")
            return {
                "status": "switched", 
                "model_type": "original",
                "features_count": len(feature_columns),
                "message": "Загружена оригинальная модель (рекомендуется для продакшена)"
            }
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки оригинальной модели: {e}")
            raise HTTPException(status_code=500, detail=f"Не удалось загрузить оригинальную модель: {e}")
    
    else:
        raise HTTPException(status_code=400, detail="Неизвестный тип модели. Используйте 'original' или 'retrained'")

@app.get("/model_status")
async def model_status():
    """Статус текущей модели"""
    return {
        "model_type": "original_v2",
        "features_count": len(feature_columns) if feature_columns else 0,
        "last_trained": last_trained,
        "model_loaded": model is not None,
        "run_id": "2f85e04263a94acbafc5aa5ed76f053c",
        "model_source": "s3://s3-student-mle-20250704-4a1b0ab232-freetrack/mlflow-artifacts/2/2f85e04263a94acbafc5aa5ed76f053c/artifacts",
        "mlflow_url": "http://localhost:5000/#/experiments/2/runs/2f85e04263a94acbafc5aa5ed76f053c",
        "metrics": {
            "ndcg_10_validation": 0.8731,
            "ndcg_10_test": 0.5079,
            "precision_5": 0.5969,
            "recall_20": 0.9934
        },
        "note": "✅ ОРИГИНАЛЬНАЯ модель v2 с 421 признаком (улучшенная версия)"
    }

@app.get("/available_models")
async def available_models():
    """Информация о доступных моделях"""
    return {
        "available_models": [
            {
                "type": "original_v2",
                "description": "ОРИГИНАЛЬНАЯ модель v2 (рекомендуется)",
                "features_count": 421,
                "run_id": "2f85e04263a94acbafc5aa5ed76f053c",
                "metrics": {
                    "ndcg_10_validation": 0.8731,
                    "ndcg_10_test": 0.5079,
                    "precision_5": 0.5969,
                    "recall_20": 0.9934
                },
                "status": "stable",
                "recommended": True
            },
            {
                "type": "original_v1", 
                "description": "Оригинальная модель v1 (устаревшая)",
                "features_count": 421,
                "run_id": "a05ede4ec398403f87ac372ae5ea1254",
                "status": "deprecated",
                "recommended": False
            },
            {
                "type": "retrained", 
                "description": "Дообученная модель (экспериментальная)",
                "features_count": "8 реальных + 413 нулевых",
                "status": "available",
                "warning": "Имеет проблемы с качеством",
                "recommended": False
            }
        ],
        "current_model": "original_v2",
        "recommendation": "✅ ИСПОЛЬЗУЙТЕ ОРИГИНАЛЬНУЮ МОДЕЛЬ v2 ДЛЯ ПРОДАКШЕНА"
    }