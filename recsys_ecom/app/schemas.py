# app/schemas.py

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class ItemFeatures(BaseModel):
    itemid: int
    features: Dict[str, Any]

class PredictRequest(BaseModel):
    visitorid: int = Field(..., description="ID пользователя")
    items: List[int] = Field(..., description="Список item_id для ранжирования")
    features: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, 
        description="Словарь фичей: {item_id: {feature_name: value}}"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "visitorid": 12345,
                "items": [1001, 1002, 1003],
                "features": {
                    "1001": {"user_prev_events": 10, "item_cum_views": 50, "hour": 14},
                    "1002": {"user_prev_events": 10, "item_cum_views": 30, "hour": 14},
                    "1003": {"user_prev_events": 10, "item_cum_views": 20, "hour": 14}
                }
            }
        }

class PredictResponse(BaseModel):
    visitorid: int
    predictions: Dict[int, float]
    ranked_items: List[int]
    
    class Config:
        schema_extra = {
            "example": {
                "visitorid": 12345,
                "predictions": {1001: 0.85, 1002: 0.72, 1003: 0.63},
                "ranked_items": [1001, 1002, 1003]
            }
        }

class RetrainResponse(BaseModel):
    status: str
    message: str
    metrics: Optional[Dict[str, float]] = None