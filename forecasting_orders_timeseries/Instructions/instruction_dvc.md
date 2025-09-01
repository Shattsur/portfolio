# Использование DVC в проекте `forecasting_orders_timeseries`

## Зачем нужен DVC

DVC (Data Version Control) — инструмент для версионирования данных .  
Код хранится в Git, а большие файлы датасеты — отдельно, но связаны с коммитами.  
Это позволяет воспроизводить эксперименты и делиться проектом без лишнего веса.

---

## Установка и настройка

1. **Активировать виртуальное окружение:**
    ```powershell
    .\venv_forecasting\Scripts\Activate.ps1
    ```

2. **Установить DVC:**
    ```powershell
    pip install dvc
    dvc --version
    ```

3. **Инициализировать DVC в проекте:**
    ```powershell
    dvc init
    git add .dvc .dvcignore .gitignore
    git commit -m "Init DVC"
    ```

---

## Добавление данных под контроль DVC

- **Сырые данные:**
  ```powershell
  dvc add db_source/taxi.csv
  git add db_source/taxi.csv.dvc .gitignore
  git commit -m "Add raw dataset taxi.csv under DVC tracking"
  ```

- **Обработанные данные:**  
  Результат ETL (`db_destination/taxi_features.csv`) отслеживается через пайплайн в `dvc.yaml`.

---

## Настройка удалённого хранилища (remote)

- **Для локального использования:**
  ```powershell
  mkdir C:\Files\Infra\dvc_storage
  dvc remote add -d local C:\Files\Infra\dvc_storage
  git add .dvc/config
  git commit -m "Configure local DVC remote storage"
  ```

- **Отправка данных в remote:**
  ```powershell
  dvc push
  ```

- **Загрузка данных из remote:**
  ```powershell
  dvc pull
  ```

---

## DVC пайплайн

Файл `dvc.yaml` описывает зависимости:

```yaml
stages:
  features:
     desc: "Фичи для прогноза заказов такси"
     cmd: powershell -Command "if (-Not (Test-Path db_destination/taxi_features.csv)) { '' | Out-File db_destination/taxi_features.csv }"
     deps:
        - db_source/taxi.csv
     outs:
        - db_destination/taxi_features.csv
```

- `deps` — входные данные (`taxi.csv`)
- `outs` — выходные данные (`taxi_features.csv`)
- `cmd` — фиктивная команда (фактическое построение фич выполняет Airflow)

---

## Работа с пайплайном

- **Проверка актуальности данных:**
  ```powershell
  dvc status
  ```

- **Пересоздание пайплайна (валидирует зависимости):**
  ```powershell
  dvc repro
  ```

---

## Версионирование и воспроизводимость

При каждом коммите фиксируется состояние данных в `dvc.lock`.

Чтобы вернуться к старой версии данных:

```powershell
git checkout <commit>
dvc pull
```

Так можно воспроизводить состояние проекта в любой момент времени.

---

## Итог

- Сырые и обработанные данные находятся под управлением DVC.
- Airflow отвечает за генерацию признаков.
- DVC обеспечивает воспроизводимость и возможность восстановления данных для любого git-коммита.