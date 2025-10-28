# tests/test_api.py
import requests
import json
import sys
import os

# Добавляем корневую директорию в путь для импортов
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def test_health():
    """Тест health endpoint"""
    print("🧪 Testing health endpoint...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        print(f"✅ Health check: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_features():
    """Тест features endpoint"""
    print("\n🧪 Testing features endpoint...")
    try:
        response = requests.get("http://localhost:8000/features", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Features count: {data['count']}")
            print(f"First 10 features: {data['features'][:10]}")
            return True
        else:
            print(f"❌ Features endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Features test failed: {e}")
        return False

def test_predict():
    """Тест predict endpoint"""
    print("\n🧪 Testing predict endpoint...")
    
    test_data = {
        "visitorid": 12345,
        "items": [1001, 1002, 1003, 1004, 1005],
        "features": {
            "1001": {
                "user_prev_events": 10,
                "user_prev_unique_items": 5,
                "item_cum_views": 150,
                "item_cum_atc": 25,
                "item_atc_rate_hist": 0.1667,
                "hour": 14,
                "day_of_week": 2,
                "is_weekend": 0,
                "prop_0": 1.5,
                "prop_1": 0.8,
                "prop_2": 2.1
            },
            "1002": {
                "user_prev_events": 10,
                "user_prev_unique_items": 5,
                "item_cum_views": 80,
                "item_cum_atc": 12,
                "item_atc_rate_hist": 0.15,
                "hour": 14,
                "day_of_week": 2,
                "is_weekend": 0,
                "prop_0": 1.2,
                "prop_1": 0.9,
                "prop_2": 1.8
            },
            "1003": {
                "user_prev_events": 10,
                "user_prev_unique_items": 5,
                "item_cum_views": 200,
                "item_cum_atc": 15,
                "item_atc_rate_hist": 0.075,
                "hour": 14,
                "day_of_week": 2,
                "is_weekend": 0,
                "prop_0": 0.9,
                "prop_1": 1.1,
                "prop_2": 2.3
            },
            "1004": {
                "user_prev_events": 10,
                "user_prev_unique_items": 5,
                "item_cum_views": 50,
                "item_cum_atc": 8,
                "item_atc_rate_hist": 0.16,
                "hour": 14,
                "day_of_week": 2,
                "is_weekend": 0,
                "prop_0": 1.8,
                "prop_1": 0.7,
                "prop_2": 1.5
            },
            "1005": {
                "user_prev_events": 10,
                "user_prev_unique_items": 5,
                "item_cum_views": 120,
                "item_cum_atc": 30,
                "item_atc_rate_hist": 0.25,
                "hour": 14,
                "day_of_week": 2,
                "is_weekend": 0,
                "prop_0": 2.1,
                "prop_1": 0.6,
                "prop_2": 1.9
            }
        }
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json=test_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Predict endpoint successful!")
            print(f"Visitor ID: {result['visitorid']}")
            print(f"Number of predictions: {len(result['predictions'])}")
            print(f"Ranked items: {result['ranked_items']}")
            
            # Показываем предсказания
            print("\n📊 Predictions:")
            for item_id, score in result['predictions'].items():
                print(f"  Item {item_id}: {score:.4f}")
                
            return True
        else:
            print(f"❌ Predict failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Predict test failed: {e}")
        return False

def main():
    """Запуск всех тестов"""
    print("🚀 Starting API tests...")
    print("=" * 50)
    
    # Проверяем, что сервис запущен
    health_ok = test_health()
    if not health_ok:
        print("\n❌ Service is not responding. Make sure it's running on localhost:8000")
        return
    
    # Запускаем остальные тесты
    features_ok = test_features()
    predict_ok = test_predict()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"Health: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"Features: {'✅ PASS' if features_ok else '❌ FAIL'}")
    print(f"Predict: {'✅ PASS' if predict_ok else '❌ FAIL'}")
    
    if all([health_ok, features_ok, predict_ok]):
        print("\n🎉 All tests passed! API is working correctly.")
    else:
        print("\n💥 Some tests failed. Check the service.")

if __name__ == "__main__":
    main()