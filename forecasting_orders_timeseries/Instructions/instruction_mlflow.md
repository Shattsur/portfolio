# Инструкция по запуску MLflow в проекте `forecasting_orders_timeseries`

## Зачем нужен MLflow

MLflow используется для:

- логирования экспериментов (метрики, гиперпараметры);
- хранения артефактов моделей;
- сравнения разных запусков моделей.

В этом проекте MLflow работает **локально**: база и артефакты хранятся в папке `mlflow` внутри проекта.

---

## Шаг 1 — установка

Активируйте виртуальное окружение проекта и установите MLflow:

```powershell
.\venv_forecasting\Scripts\Activate.ps1
pip install mlflow
mlflow --version
```

---

## Шаг 2 — структура папок

В корне проекта создаётся каталог для MLflow:

```
forecasting_orders_timeseries/
│
├── mlflow/
│   ├── mlflow.db       # база данных SQLite (создаётся автоматически)
│   └── artifacts/      # хранилище артефактов моделей
```

---

## Шаг 3 — запуск MLflow Tracking Server

Можно запустить вручную:

```powershell
mlflow server `
    --backend-store-uri sqlite:///mlflow/mlflow.db `
    --default-artifact-root file:/C:/Files/Обучение/Projects/portfolio/forecasting_orders_timeseries/mlflow/artifacts `
    --host 127.0.0.1 `
    --port 5000
```

После запуска интерфейс доступен по адресу:  
👉 http://127.0.0.1:5000

---

## Шаг 4 — запуск через PowerShell-скрипт

В проекте можно использовать скрипт `run_mlflow.ps1`:

```powershell
.\run_mlflow.ps1
```

Этот скрипт создаёт папки (если их нет) и запускает сервер на http://127.0.0.1:5000.

---

## Шаг 5 — логирование экспериментов

Пример Python-кода:

```python
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("taxi_forecasting")

with mlflow.start_run(run_name="xgboost_baseline"):
    mlflow.log_param("max_depth", 6)
    mlflow.log_metric("rmse", 38.3)
    mlflow.log_metric("r2", 0.577)
```

После запуска кода метрики будут отображаться в интерфейсе MLflow.

---

## Шаг 6 — завершение работы

Остановить сервер можно сочетанием `Ctrl + C` в консоли, где он запущен.  

---

## Итог

- MLflow поднимается локально в папке `mlflow`.
- Веб-интерфейс доступен на http://127.0.0.1:5000.

Загрузка модели

```python
model = mlflow.xgboost.load_model(f"runs:/{run_id}/model")  # XGBModel
predictions = model.predict(X_test)  
predictions[:5]
```