import random
import numpy as np
from PIL import Image, ImageFile, ImageEnhance
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from transformers import AutoTokenizer, AutoImageProcessor
from functools import partial
from albumentations.pytorch import ToTensorV2
import pandas as pd

ImageFile.LOAD_TRUNCATED_IMAGES = True

def compute_target_statistics(df, target_column="total_calories"):
    """Вычисляет статистику целевой переменной из DataFrame"""
    calories = df[target_column].values
    mean = float(np.mean(calories))
    std = float(np.std(calories))
    
    print(f"📊 Вычислена статистика целевой переменной:")
    print(f"   • MEAN: {mean:.2f}")
    print(f"   • STD: {std:.2f}")
    print(f"   • Диапазон: [{calories.min():.1f}, {calories.max():.1f}]")
    print(f"   • Количество образцов: {len(calories)}")
    
    return mean, std

# ==========================
# === ПРОСТЫЕ ТЕКСТОВЫЕ АУГМЕНТАЦИИ БЕЗ NLTK ===
# ==========================
def simple_text_augmentation(text):
    if len(text.split()) <= 2:
        return text
    words = text.split()
    if random.random() < 0.3 and len(words) > 2:
        random.shuffle(words)
        return ' '.join(words)
    if random.random() < 0.2 and len(words) > 3:
        del_idx = random.randint(0, len(words)-1)
        words.pop(del_idx)
        return ' '.join(words)
    ingredient_replacements = {
        'tomato': ['tomatoes', 'fresh tomato'],
        'onion': ['onions', 'red onion'], 
        'cheese': ['grated cheese', 'shredded cheese'],
        'chicken': ['chicken breast', 'chicken meat'],
        'beef': ['beef meat', 'ground beef'],
        'oil': ['cooking oil', 'vegetable oil'],
        'salt': ['sea salt', 'table salt'],
        'pepper': ['black pepper', 'ground pepper'],
        'garlic': ['fresh garlic', 'garlic cloves'],
        'butter': ['melted butter', 'unsalted butter']
    }
    new_words = []
    for word in words:
        word_lower = word.lower().strip('.,!?')
        if word_lower in ingredient_replacements and random.random() < 0.3:
            new_words.append(random.choice(ingredient_replacements[word_lower]))
        else:
            new_words.append(word)
    return ' '.join(new_words)

def random_ingredient_dropout(text, p=0.15):
    if ',' in text:
        ingredients = [ing.strip() for ing in text.split(',')]
        kept = [ing for ing in ingredients if random.random() > p and ing]
        return ', '.join(kept) if kept else text
    else:
        words = text.split()
        if len(words) <= 2:
            return text
        kept = [word for word in words if random.random() > p]
        return ' '.join(kept) if kept else text

# ==========================
# === ИЗОБРАЖЕНИЯ ===
# ==========================
def get_transforms(model_name, size, mode="train"):
    if mode == "train":
        return A.Compose([
            A.Resize(size, size),
            A.HorizontalFlip(p=0.3),
            A.RandomBrightnessContrast(0.1, 0.1, p=0.2),
            A.OneOf([
                A.GaussianBlur(3, p=0.1),
                A.MotionBlur(3, p=0.1),
            ], p=0.2),
        ])
    else:
        return A.Compose([A.Resize(size, size)])

# ==========================
# === DATASET С ЧИСЛОВЫМИ ПРИЗНАКАМИ И СТАТИСТИКОЙ ===
# ==========================
class MultimodalDataset(Dataset):
    def __init__(self, df, transforms, text_augment=False, config=None, target_mean=None, target_std=None):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.text_augment = text_augment
        self.config = config
        
        # Сохраняем статистику целевой переменной
        self.target_mean = target_mean
        self.target_std = target_std
        
        # Вычисляем статистики для нормализации числовых признаков
        if config and getattr(config, 'USE_NUMERIC_FEATURES', False):
            self._compute_numeric_stats(df)
    
    def _compute_numeric_stats(self, df):
        """Вычисляем mean и std ТОЛЬКО для n_ingredients"""
        self.ingr_mean = df["n_ingredients"].mean() 
        self.ingr_std = df["n_ingredients"].std()
        
        print(f"📊 Numeric features stats:")
        print(f"   n_ingredients: mean={self.ingr_mean:.1f}, std={self.ingr_std:.1f}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["ingredients_text"])
        label = float(row["total_calories"])
        img_path = row["image_path"]

        # Текстовые аугментации
        if self.text_augment and random.random() < 0.6:
            text = simple_text_augmentation(text)

        # Изображение
        try:
            pil = Image.open(img_path).convert("RGB")
            img = np.array(pil)
            img = self.transforms(image=img)["image"]
            
            # Дополнительная случайная коррекция яркости/цвета
            if self.text_augment and random.random() < 0.3:
                pil_aug = Image.fromarray((img).astype(np.uint8))
                pil_aug = ImageEnhance.Sharpness(pil_aug).enhance(random.uniform(0.8,1.5))
                pil_aug = ImageEnhance.Color(pil_aug).enhance(random.uniform(0.9,1.2))
                img = np.array(pil_aug).astype(np.float32)
                
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            img = np.zeros((self.transforms.height if hasattr(self.transforms,'height') else 224,
                            self.transforms.width if hasattr(self.transforms,'width') else 224, 3), dtype=np.float32)

        # Числовые признаки (если включены в конфиге) - ТОЛЬКО n_ingredients
        numeric_features = None
        if self.config and getattr(self.config, 'USE_NUMERIC_FEATURES', False):
            ingr_norm = (row["n_ingredients"] - self.ingr_mean) / self.ingr_std
            numeric_features = torch.tensor([ingr_norm], dtype=torch.float32)

        return {
            "text": text, 
            "image": img, 
            "label": label, 
            "idx": idx,
            "numeric": numeric_features
        }

# ==========================
# === COLLATE FUNCTION С ЧИСЛОВЫМИ ПРИЗНАКАМИ ===
# ==========================
def collate_fn(batch, tokenizer, image_processor, max_length=256, config=None):
    texts = ["passage: " + str(b["text"]) for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
    indices = [b["idx"] for b in batch]

    # Токенизация текста
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True
    )

    # Подготовка изображений к image_processor
    images = [b["image"] for b in batch]  
    pixel_batch = image_processor(images=images, return_tensors="pt")
    images_tensor = pixel_batch["pixel_values"]

    # Числовые признаки (если есть)
    numeric_features = None
    if config and getattr(config, 'USE_NUMERIC_FEATURES', False):
        numeric_list = [b["numeric"] for b in batch if b["numeric"] is not None]
        if numeric_list:
            numeric_features = torch.stack(numeric_list)

    result = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "image": images_tensor,
        "label": labels,
        "indices": indices
    }
    
    # Добавляем числовые признаки если они есть
    if numeric_features is not None:
        result["numeric"] = numeric_features

    return result

# ==========================
# === CREATE DATALOADERS С ВЫЧИСЛЕНИЕМ СТАТИСТИКИ ===
# ==========================

def create_dataloaders(train_df, val_df, config):
    # Вычисляем статистику целевой переменной на тренировочных данных
    target_mean, target_std = compute_target_statistics(train_df)
    
    # Обновляем конфиг вычисленными значениями
    config.TARGET_MEAN = target_mean
    config.TARGET_STD = target_std
    
    # Автоматическое определение использования быстрого токенизатора
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL, use_fast=True)
        print("✅ Using fast tokenizer")
    except (ValueError, TypeError) as e:
        print(f"⚠️ Fast tokenizer failed: {e}. Falling back to slow tokenizer.")
        tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL, use_fast=False)
    
    # Устанавливаем pad_token если его нет
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
    
    image_processor = AutoImageProcessor.from_pretrained(config.IMAGE_MODEL)

    # Создаем даталоадеры с передачей вычисленной статистики
    train_loader = DataLoader(
        MultimodalDataset(
            train_df,
            get_transforms(config.IMAGE_MODEL, config.IMAGE_SIZE, "train"),
            text_augment=True,
            config=config,
            target_mean=target_mean,
            target_std=target_std
        ),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=partial(collate_fn, 
                          tokenizer=tokenizer, 
                          image_processor=image_processor, 
                          max_length=config.MAX_SEQ_LENGTH,
                          config=config),
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        MultimodalDataset(
            val_df,
            get_transforms(config.IMAGE_MODEL, config.IMAGE_SIZE, "eval"),
            text_augment=False,
            config=config,
            target_mean=target_mean,  # Используем ту же статистику для валидации!
            target_std=target_std     # Важно: используем train статистику для val
        ),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=partial(collate_fn, 
                          tokenizer=tokenizer, 
                          image_processor=image_processor, 
                          max_length=config.MAX_SEQ_LENGTH,
                          config=config),
        num_workers=0,
        pin_memory=True
    )

    print(f"✅ DataLoaders created: Train {len(train_df)} samples, Val {len(val_df)} samples")
    
    # Информация о числовых признаках
    if getattr(config, 'USE_NUMERIC_FEATURES', False):
        print(f"📊 Using numeric features: n_ingredients")
    
    return train_loader, val_loader, tokenizer, image_processor