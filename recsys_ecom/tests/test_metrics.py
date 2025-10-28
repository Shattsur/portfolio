# test_metrics.py для отображения метрик в реальном времени
import requests
import time
import random
import threading

def generate_predict_traffic():
    """Генерация предсказаний"""
    base_url = "http://localhost:8000"
    test_items = [1001, 1002, 1003, 1004, 1005]
    
    for i in range(100):  # Больше запросов
        try:
            visitor_id = random.randint(1000, 9999)
            
            features = {}
            for item_id in test_items:
                features[str(item_id)] = {
                    "user_prev_events": random.randint(1, 100),
                    "user_prev_unique_items": random.randint(1, 20),
                    "item_cum_views": random.randint(10, 1000),
                    "item_cum_atc": random.randint(0, 50),
                    "item_atc_rate_hist": random.uniform(0, 0.1),
                    "hour": random.randint(0, 23),
                    "day_of_week": random.randint(0, 6),
                    "is_weekend": random.randint(0, 1)
                }
            
            response = requests.post(
                f"{base_url}/predict",
                json={
                    "visitorid": visitor_id,
                    "items": test_items,
                    "features": features
                },
                timeout=5
            )
            
            if response.status_code == 200:
                print(f"✅ Prediction {i+1}: Success")
            else:
                print(f"❌ Prediction {i+1}: Failed - {response.status_code}")
                
        except Exception as e:
            print(f"💥 Prediction {i+1}: Error - {e}")
        
        time.sleep(random.uniform(0.1, 1.0))  # Случайные интервалы

def generate_health_checks():
    """Генерация health check запросов"""
    base_url = "http://localhost:8000"
    
    for i in range(50):
        try:
            response = requests.get(f"{base_url}/health", timeout=2)
            if response.status_code == 200:
                print(f"🔍 Health check {i+1}: OK")
            else:
                print(f"⚠️ Health check {i+1}: {response.status_code}")
        except Exception as e:
            print(f"💥 Health check {i+1}: Error - {e}")
        
        time.sleep(random.uniform(0.5, 2.0))

def generate_features_requests():
    """Запросы информации о фичах"""
    base_url = "http://localhost:8000"
    
    for i in range(20):
        try:
            response = requests.get(f"{base_url}/features", timeout=3)
            if response.status_code == 200:
                print(f"📊 Features request {i+1}: OK")
            else:
                print(f"❌ Features request {i+1}: {response.status_code}")
        except Exception as e:
            print(f"💥 Features request {i+1}: Error - {e}")
        
        time.sleep(random.uniform(1.0, 3.0))

if __name__ == "__main__":
    print("🚀 Generating comprehensive test traffic for metrics...")
    
    # Запускаем в разных потоках для имитации реальной нагрузки
    threads = []
    
    threads.append(threading.Thread(target=generate_predict_traffic))
    threads.append(threading.Thread(target=generate_health_checks))
    threads.append(threading.Thread(target=generate_features_requests))
    
    for thread in threads:
        thread.start()
    
    for thread in threads:
        thread.join()
    
    print("✅ Test traffic generation completed!")