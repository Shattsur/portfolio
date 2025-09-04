# run_mlflow.ps1
# Скрипт для запуска локального MLflow Tracking Server

# Путь к проекту
$ProjectDir = "C:\Files\AI\Projects\portfolio\forecasting_orders_timeseries"
$MlflowDir  = "$ProjectDir\mlflow"

# Создать папки, если их нет
if (!(Test-Path "$MlflowDir\artifacts")) {
    New-Item -ItemType Directory -Path "$MlflowDir\artifacts" -Force | Out-Null
}

# Запуск MLflow Tracking Server
mlflow server `
    --backend-store-uri "sqlite:///$MlflowDir/mlflow.db" `
    --default-artifact-root "file:///$MlflowDir/artifacts" `
    --host 127.0.0.1 `
    --port 5000