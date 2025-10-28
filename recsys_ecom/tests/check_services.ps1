# check_services.ps1

Write-Host "CHECKING RECSYS SERVICES" -ForegroundColor Green
Write-Host "=========================" -ForegroundColor Green
Write-Host ""

# 1. Check container status
Write-Host "1. CONTAINER STATUS:" -ForegroundColor Yellow
docker-compose ps
Write-Host ""

# 2. Check logs for each service (last 20 lines)
Write-Host "2. SERVICE LOGS (last 20 lines):" -ForegroundColor Yellow
Write-Host ""

$services = @(
    @{Name="postgres"; Description="Database"},
    @{Name="airflow-webserver"; Description="Airflow Web Interface"},
    @{Name="airflow-scheduler"; Description="Airflow Scheduler"},
    @{Name="mlflow"; Description="MLflow Tracking"},
    @{Name="recsys_api"; Description="RecSys API"}
)

foreach ($service in $services) {
    Write-Host "--- $($service.Description) ($($service.Name)) ---" -ForegroundColor Cyan
    try {
        $logs = docker-compose logs --tail=20 $service.Name 2>&1
        if ($LASTEXITCODE -eq 0) {
            $logs
        } else {
            Write-Host "Error getting logs for $($service.Name)" -ForegroundColor Red
        }
    } catch {
        Write-Host "Failed to get logs for $($service.Name)" -ForegroundColor Red
    }
    Write-Host ""
}

# 3. Check service health via HTTP
Write-Host "3. HTTP HEALTH CHECK:" -ForegroundColor Yellow
Write-Host ""

$endpoints = @(
    @{Url="http://localhost:8080/health"; Service="Airflow"},
    @{Url="http://localhost:8000/health"; Service="RecSys API"},
    @{Url="http://localhost:5000"; Service="MLflow"}
)

foreach ($endpoint in $endpoints) {
    Write-Host "Checking $($endpoint.Service) ($($endpoint.Url))..." -NoNewline
    try {
        $response = Invoke-WebRequest -Uri $endpoint.Url -TimeoutSec 10 -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            Write-Host " AVAILABLE" -ForegroundColor Green
            if ($endpoint.Service -eq "RecSys API") {
                $content = $response.Content | ConvertFrom-Json
                Write-Host "   Status: $($content.status)" -ForegroundColor Green
                Write-Host "   Model loaded: $($content.model_loaded)" -ForegroundColor Green
            }
        } else {
            Write-Host " ERROR: $($response.StatusCode)" -ForegroundColor Red
        }
    } catch {
        Write-Host " UNAVAILABLE" -ForegroundColor Red
        Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host ""

# 4. Check DAGs in Airflow
Write-Host "4. CHECKING AIRFLOW DAGS:" -ForegroundColor Yellow
try {
    $dags = docker exec mle-pr-final-airflow-webserver-1 airflow dags list 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Available DAGs:" -ForegroundColor Green
        $dags | Select-String -Pattern "recsys" | ForEach-Object {
            Write-Host "   $($_.Line.Trim())" -ForegroundColor Green
        }
    } else {
        Write-Host "Failed to get DAG list" -ForegroundColor Red
    }
} catch {
    Write-Host "Failed to check DAGs" -ForegroundColor Red
}

Write-Host ""

# 5. Check network
Write-Host "5. NETWORK CHECK:" -ForegroundColor Yellow
try {
    $network = docker network inspect mle-pr-final_recsys-network --format "{{.Name}}: {{.Driver}}"
    Write-Host "Network: $network" -ForegroundColor Green
    
    $containers = docker network inspect mle-pr-final_recsys-network --format "{{range .Containers}}{{.Name}} {{end}}"
    Write-Host "Containers in network: $containers" -ForegroundColor Green
} catch {
    Write-Host "Error checking network" -ForegroundColor Red
}

Write-Host ""
Write-Host "=========================" -ForegroundColor Green
Write-Host "CHECK COMPLETED" -ForegroundColor Green