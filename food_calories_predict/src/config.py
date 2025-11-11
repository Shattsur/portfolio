# config.py

from pathlib import Path
import torch

class Config:
    # Пути
    BASE_DIR = Path(".")
    SAVE_PATH = BASE_DIR / "best_model.pth"
    IMAGE_DIR = BASE_DIR / "images"
    
    
    
    TEXT_MODEL = "BAAI/bge-large-en-v1.5"
    
    IMAGE_MODEL = "facebook/dinov2-with-registers-base"
        
    USE_NUMERIC_FEATURES = True  # Включить/выключить числовые признаки
    NUMERIC_FEATURES = ["n_ingredients"]
    NUMERIC_DIM = 32  # Размерность для проекции числовых признаков

    # Архитектура
    HIDDEN_DIM = 512  
    DROPOUT = 0.3    
    DROPOUT_FACTORS = {
        "text": 0.5,
        "image": 0.5,
        "numeric": 0.3,
        "regressor": 1.0
    }

    # Токенизация текста
    MAX_SEQ_LENGTH = 128  

    # Планировщик обучения    
    WARMUP_EPOCHS = 5
    MIN_LR = 1e-6
    
    # Обучение
    SEED = 42
    BATCH_SIZE = 32  
    EPOCHS = 100      

    # Learning Rates 
    LR_TEXT = 5e-5
    LR_IMAGE = 5e-5  
    LR_HEAD = 2e-4     
    
    # Увеличить общее терпение для early stopping
    PATIENCE = 12
       
    # Дополнительные параметры
    LOG_INTERVAL = 1      
    
    # Регуляризация
    WEIGHT_DECAY = 0.03   
    GRAD_CLIP_NORM = 1.0 
    
   
    # Функция потерь
    LOSS_TYPE = "huber"  # "mse", "huber", "weighted_mse", "combined"
    
    # Статистика будет автоматически вычислена в create_dataloaders
    # на основе тренировочных данных
    TARGET_MEAN = None  # Будет вычислено в create_dataloaders
    TARGET_STD = None   # Будет вычислено в create_dataloaders

    TEXT_UNFREEZE = r"encoder\.layer\.(16|17|18|19|20|21|22|23)" 
    
    IMAGE_UNFREEZE = r"encoder\.layer\.(7|8|9|10|11)|layernorm|register_tokens"
  

    # Данные
    IMAGE_SIZE = 224 
    
    # Устройство
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

config = Config()