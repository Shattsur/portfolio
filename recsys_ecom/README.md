# 🧠 RecSys Ranker

**Производственная система ранжирования товаров на базе LightGBM (LambdaRank)** с полноценным **MLOps-контуром**: от подготовки событийных данных и автоматического поиска гиперпараметров до деплоя модели, мониторинга и оркестрации переобучения.

Система ориентирована на продакшен-флоу — **версионирование моделей и фич**, **логирование экспериментов в MLflow**, **хранение артефактов в S3**, **наблюдаемость через Prometheus/Grafana** и **безопасное переключение моделей** в runtime через API.  
Подходит для регулярного **автоматического переобучения (Airflow DAGs)**, ручного дообучения новых версий модели.

---

## 🚩 Ключевые свойства

- **Повторяемость:** обучение и артефакты фиксируются в MLflow и S3.  
- **Контроль качества:** Optuna + валидация NDCG@k + ручная/автоматическая проверка перед переключением.  
- **Наблюдаемость:** метрики в Prometheus → Grafana, логи в Docker.  
- **Безопасность эксплуатации:** healthchecks, ручное переключение моделей, изоляция сервисов в Docker Compose.

---

## 🧩 Стек технологий

| Компонент | Назначение |
|------------|-------------|
| **LightGBM (LambdaRank)** | модель ранжирования |
| **Optuna** | TPE-поиск гиперпараметров |
| **MLflow** | трекинг, артефакты, Registry (S3) |
| **FastAPI + Pydantic** | REST API для инференса и управления |
| **Airflow** | оркестрация регулярного дообучения |
| **PostgreSQL** | metadata DB (Airflow, MLflow при необходимости) |
| **S3 / Yandex Object Storage** | хранилище артефактов |
| **Prometheus + Grafana** | мониторинг и дашборды |
| **Docker + Compose** | деплой окружения |
| **boto3 / pytest** | интеграция с S3, автотесты |

---

## 🚀 Быстрый старт

### Требования

- Docker & Docker Compose  
- ~4 GB свободной памяти  
- Доступ к S3-совместимому хранилищу (например, Yandex Cloud S3)

### Установка и запуск

```bash
git clone <repo_url>
cd <project_root>
cp .env.example .env
docker-compose up -d --build
docker-compose ps
# завершение работы 
docker-compose down (при ошибке повторной сборки - docker-compose down -v --remove-orphans)
```
Для дообучения модели запустите в Airflow DAG "recsys_real_retraining", новая модель будет сохранена в S3.
По умолчанию используется оригинальная модель, для использования в дальнейшем дообученной версии, воспользуйтесь функцией API switch.
---

## 🌐 Интерфейсы

| Сервис | URL | Логин / Пароль |
|---------|-----|----------------|
| **API & Docs** | http://localhost:8000/docs | — |
| **MLflow** | http://localhost:5000 | — |
| **Airflow** | http://localhost:8080 | admin / admin |
| **Grafana** | http://localhost:3000 | admin / admin |
| **Prometheus** | http://localhost:9090 | — |


---

## 🐳 Работа с Docker

```bash
docker-compose ps
docker-compose logs -f recsys_api
docker-compose exec recsys_api /bin/bash
docker inspect --format='{{json .State.Health}}' $(docker-compose ps -q recsys_api)
docker-compose down -v
```

---

## 🔗 Основные эндпоинты API

| Метод | Endpoint | Описание |
|-------|---------|----------|
| GET   | `/health` | Проверка состояния сервиса и модели |
| GET   | `/features` | Получение списка используемых признаков и их количества |
| POST  | `/predict` | Получение предсказаний и ранжированного списка товаров |
| POST  | `/retrain` | Запуск фонового дообучения модели |
| POST  | `/switch_model?model_type={original|retrained}` | Переключение между оригинальной и дообученной моделью |
| GET   | `/metrics` | Экспонирование метрик Prometheus для мониторинга |

---

## 🔁 Пайплайны

1️⃣ **Обучение + Optuna → MLflow → S3**  
2️⃣ **Дообучение через API → MLflow → S3**  
3️⃣ **Airflow DAG — recsys_real_retraining**

---

## ⚙️ Конфигурация .env

```bash
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
S3_BUCKET_NAME=recsys-models
MLFLOW_S3_ENDPOINT_URL=https://storage.yandexcloud.net  
DB_DESTINATION_HOST=postgres
DB_DESTINATION_PORT=5432
DB_DESTINATION_NAME=airflow
DB_DESTINATION_USER=airflow
DB_DESTINATION_PASSWORD=airflow
DATA_DIR=./data
EVENTS_FILE=events.parquet
MLFLOW_TRACKING_URI=http://mlflow:5000
PYTHONUNBUFFERED=1
LOG_LEVEL=INFO
```

---

## 📡 Мониторинг

Система мониторинга построена на связке **Prometheus + Grafana**, обеспечивая полную наблюдаемость за API и моделью.

### 🔍 Prometheus
Prometheus собирает метрики с сервисов через endpoint `/metrics` FastAPI (экспонируется библиотекой `prometheus_client`).

### 📊 Grafana
Grafana автоматически подхватывает дашборды через provisioning, используя JSON-конфигурации, размещённые в `/etc/grafana/provisioning/dashboards`.
Для генерации тестового трафика воспользуйтесь test_metrics.py.

---

### 🔢 Ключевые PromQL-запросы

| **Метрика** | **PromQL** | **Назначение** |
|--------------|-------------|----------------|
| **API Rate** | `rate(api_requests_total[5m])` | скорость запросов к API |
| **Prediction Rate** | `rate(model_predictions_total[5m])` | количество предсказаний в минуту |
| **p95 Latency** | `histogram_quantile(0.95, sum(rate(model_prediction_duration_seconds_bucket[5m])) by (le))` | 95-й перцентиль задержки |
| **Success Rate** | `rate(model_predictions_total{status="success"}[5m]) / rate(model_predictions_total[5m])` | доля успешных запросов |
| **Model Load Status** | `model_load_status` | индикатор загруженной модели (0/1) |

---

Дашборд **"RecSys Production Monitoring"** включает ключевые панели:
- *API Requests Rate* — частота запросов к API.
- *Model Predictions Rate* — динамика инференсов.
- *Prediction Latency (p95)* — контроль производительности модели.
- *Success Rate* — надёжность ответов модели.
- *Model Load Status* — текущий статус загруженной модели.

Все панели обновляются каждые **10 секунд**, обеспечивая real-time наблюдаемость.

---

## 🧪 Тестирование

Система включает полный набор тестов для проверки корректности работы всех компонентов RecSys Ranker — от инфраструктуры до API и метрик.

---

### ⚙️ **Интеграционные тесты**

**`check_services.ps1`**  
PowerShell-скрипт для комплексной проверки продакшен-среды:
- статус Docker-контейнеров (`docker-compose ps`);
- последние логи сервисов (Airflow, MLflow, RecSys API);
- HTTP health-check каждого сервиса;
- доступность DAG'ов в Airflow;
- состояние сети Docker.

Запуск:
```bash
powershell ./tests/check_services.ps1
```

### 🧩 Тесты MLflow

**`tests/mlflow_test.py`**  
Проверяет доступность **MLflow Tracking Server** и корректность API:

- `/health` — проверка состояния сервера;  
- `/api/2.0/mlflow/runs/search` — поиск экспериментов.

---

### 🚀 Тесты API

**`tests/test_api.py`**  
Пошаговая проверка REST API:

- `/health` — доступность сервиса;  
- `/features` — корректность и количество признаков;  
- `/predict` — ранжирование товаров и формат предсказаний.

---

### 📊 Генерация тестового трафика

**`tests/test_metrics.py`**  
Создаёт имитацию реальной нагрузки на API:

- множественные запросы к `/predict`, `/features`, `/health`;  
- позволяет наблюдать метрики в реальном времени через **Prometheus + Grafana**.

---

### 🤖 Тест предсказаний

**`tests/test_predictions.py`**  
Отправляет тестовые запросы на `/predict`, анализирует ранжирование и значения скорингов, проверяя целостность входных данных и корректность модели.

---

🧭 Все тесты можно запускать как локально, так и внутри контейнеров.  
Они обеспечивают проверку ключевых аспектов продакшен-контура: **доступность, стабильность и точность модели**.


### 🎯 Рекомендации

| Метрика | Цель |
|----------|------|
| NDCG@10 | ≥ 0.75 |
| Precision@5 | ≥ 0.60 |
| Recall@20 | ≥ 0.85 |
| p95 latency | < 100 ms |

---

© 2025 RecSys Ranker — production-grade ML ranking system with full MLOps lifecycle.
