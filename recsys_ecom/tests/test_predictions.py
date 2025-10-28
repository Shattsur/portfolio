# test_predictions.py
import requests
import json

def test_predictions():
    """Тестирование предсказаний API"""
    
    # 1. Проверка здоровья
    health_response = requests.get("http://localhost:8000/health")
    print("🔍 Health Check:", health_response.json())
    
    # 2. Проверка фичей
    features_response = requests.get("http://localhost:8000/features")
    features = features_response.json()
    print(f"📊 Доступно фичей: {features['count']}")
    print("Первые 10 фичей:", features['features'][:10])
    
    # 3. Тестовое предсказание
    test_data = {
        "visitorid": 12345,
        "items": [1001, 1002, 1003, 1004, 1005],
        "features": {
            "1001": {
                "user_prev_events": 10,
                "user_prev_unique_items": 5, 
                "item_cum_views": 100,
                "item_cum_atc": 5,
                "item_atc_rate_hist": 0.05,
                "hour": 14,
                "day_of_week": 2,
                "is_weekend": 0
            },
            "1002": {
                "user_prev_events": 10,
                "user_prev_unique_items": 5,
                "item_cum_views": 50, 
                "item_cum_atc": 3,
                "item_atc_rate_hist": 0.06,
                "hour": 14,
                "day_of_week": 2,
                "is_weekend": 0
            },
            "1003": {
                "user_prev_events": 10,
                "user_prev_unique_items": 5,
                "item_cum_views": 200,
                "item_cum_atc": 8, 
                "item_atc_rate_hist": 0.04,
                "hour": 14,
                "day_of_week": 2,
                "is_weekend": 0
            },
            "1004": {
                "user_prev_events": 10,
                "user_prev_unique_items": 5,
                "item_cum_views": 75,
                "item_cum_atc": 2,
                "item_atc_rate_hist": 0.026,
                "hour": 14,
                "day_of_week": 2, 
                "is_weekend": 0
            },
            "1005": {
                "user_prev_events": 10,
                "user_prev_unique_items": 5,
                "item_cum_views": 150,
                "item_cum_atc": 6,
                "item_atc_rate_hist": 0.04,
                "hour": 14,
                "day_of_week": 2,
                "is_weekend": 0
            }
        }
    }
    
    # Заполняем остальные фичи нулями (модель сама дополнит)
    for item_id in test_data["items"]:
        for i in range(400):
            prop_key = f"prop_{i}"
            if prop_key not in test_data["features"][str(item_id)]:
                test_data["features"][str(item_id)][prop_key] = 0.0
    
    # 4. Делаем предсказание
    predict_response = requests.post(
        "http://localhost:8000/predict",
        json=test_data
    )
    
    if predict_response.status_code == 200:
        result = predict_response.json()
        print("✅ Предсказание успешно!")
        print(f"👤 Visitor ID: {result['visitorid']}")
        print("📊 Predictions:")
        for item_id, score in result['predictions'].items():
            print(f"   - Item {item_id}: {score:.4f}")
        print(f"🏆 Ранжированный список: {result['ranked_items']}")
    else:
        print(f"❌ Ошибка предсказания: {predict_response.status_code}")
        print(predict_response.text)

if __name__ == "__main__":
    test_predictions()