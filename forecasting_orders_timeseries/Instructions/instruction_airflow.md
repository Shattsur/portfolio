# Инструкция по запуску Airflow в проекте `forecasting_orders_timeseries`

## Шаг 1 — Задать переменные окружения

Создайте файл `airflow/.env` со следующим содержимым:

```
SOURCE_PATH=/opt/airflow/db_source/taxi.csv
DEST_PATH=/opt/airflow/db_destination/taxi_features.csv
```

> **Важно:** Пути должны совпадать с теми, что указаны в `air_etl.py`.

---

## Шаг 2 — Проверить `docker-compose.yaml`

Убедитесь, что в `airflow/docker-compose.yaml` примонтированы каталоги с данными и DAG-ами:

```yaml
services:
  airflow-webserver:
    volumes:
      - ../db_source:/opt/airflow/db_source
      - ../db_destination:/opt/airflow/db_destination
      - ./dags:/opt/airflow/dags
      - ./logs:/opt/airflow/logs
      - ./plugins:/opt/airflow/plugins
    env_file:
      - .env
```

---

## Шаг 3 — Сборка и запуск контейнеров

Выполните в директории `airflow/`:

```powershell
cd airflow
docker-compose up --build -d
```

Airflow поднимет следующие сервисы:

- webserver (UI: http://localhost:8080)
- scheduler
- postgres
- другие вспомогательные контейнеры

---

## Шаг 4 — Доступ в интерфейс

Перейдите в браузере по адресу: [http://localhost:8080](http://localhost:8080)

Логин/пароль по умолчанию (или из переменных окружения):

```
login: airflow
password: airflow
```

---

## Шаг 5 — Запуск DAG

1. Найдите DAG с именем `air_etl` в интерфейсе Airflow.
2. Включите тумблер (Activate DAG).
3. Нажмите **Trigger DAG** для ручного запуска.

Airflow выполнит три задачи:

- **extract** — загрузка данных из `db_source/taxi.csv`
- **transform** — ресемплинг и генерация признаков
- **load** — сохранение результата в `db_destination/taxi_features.csv`

---

## Шаг 6 — Проверка результатов

После выполнения DAG в папке `db_destination` появится обновлённый файл:

```
db_destination/taxi_features.csv
```