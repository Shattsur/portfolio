# model.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoImageProcessor

# --- mean pooling ---
def mean_pooling(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(1)
    counts = mask.sum(1).clamp(min=1e-9)
    return summed / counts

class MultimodalModel(nn.Module):
    """
    Улучшенная мультимодальная модель с поддержкой числовых признаков:
    - Текст и изображение проецируются в bottleneck-пространство
    - Только n_ingredients добавляется через отдельную проекцию
    - Конкатенация всех модальностей + регрессор
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # === TEXT ===
        text_config = AutoConfig.from_pretrained(config.TEXT_MODEL)
        self.text_model = AutoModel.from_pretrained(config.TEXT_MODEL, config=text_config)

        # === IMAGE ===
        self.image_processor = AutoImageProcessor.from_pretrained(config.IMAGE_MODEL)
        self.image_model = AutoModel.from_pretrained(config.IMAGE_MODEL)

        # Bottleneck размерность
        bottleneck_dim = 384
        
        # Простые bottleneck проекции
        self.text_bottleneck = nn.Sequential(
            nn.Linear(self.text_model.config.hidden_size, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Dropout(config.DROPOUT * config.DROPOUT_FACTORS["text"])
        )
        
        self.image_bottleneck = nn.Sequential(
            nn.Linear(self.image_model.config.hidden_size, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Dropout(config.DROPOUT * config.DROPOUT_FACTORS["image"])
        )

        # Проекция для числовых признаков (если используются)
        if getattr(config, 'USE_NUMERIC_FEATURES', False):
            # Теперь только 1 признак - n_ingredients
            self.numeric_bottleneck = nn.Sequential(
                nn.Linear(1, getattr(config, 'NUMERIC_DIM', 32)),  # ← Только 1 вход!
                nn.LayerNorm(getattr(config, 'NUMERIC_DIM', 32)),
                nn.GELU(),
                nn.Dropout(config.DROPOUT * config.DROPOUT_FACTORS["numeric"])
            )
            # Обновляем входную размерность регрессора
            regressor_input_dim = bottleneck_dim * 2 + getattr(config, 'NUMERIC_DIM', 32)
        else:
            self.numeric_bottleneck = None
            regressor_input_dim = bottleneck_dim * 2

        # Регрессор с поддержкой числовых признаков
        self.regressor = nn.Sequential(
            nn.Linear(regressor_input_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(config.DROPOUT * config.DROPOUT_FACTORS["regressor"]),
            nn.Linear(bottleneck_dim, 1)
        )

    def forward(self, input_ids, attention_mask, images, numeric_features=None):
        """
        input_ids, attention_mask: тензоры для text_model
        images: тензор pixel_values  
        numeric_features: тензор числовых признаков [n_ingredients_norm] - форма (B, 1)
        """
        # --- TEXT ---
        txt_out = self.text_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        text_pooled = mean_pooling(txt_out.last_hidden_state, attention_mask)
        text_emb = self.text_bottleneck(text_pooled)  # (B, bottleneck_dim)

        # --- IMAGE ---
        img_out = self.image_model(images)
        if hasattr(img_out, "last_hidden_state"):
            image_pooled = img_out.last_hidden_state.mean(dim=1)
        else:
            image_pooled = img_out.pooler_output
        image_emb = self.image_bottleneck(image_pooled)  # (B, bottleneck_dim)

        # --- NUMERIC FEATURES (если есть) ---
        if numeric_features is not None and self.numeric_bottleneck is not None:
            # numeric_features имеет форму (B, 1) - только n_ingredients
            numeric_emb = self.numeric_bottleneck(numeric_features)  # (B, NUMERIC_DIM)
            # Конкатенируем все модальности
            fused = torch.cat([text_emb, image_emb, numeric_emb], dim=1)  # (B, bottleneck_dim * 2 + NUMERIC_DIM)
        else:
            # Только текст и изображение
            fused = torch.cat([text_emb, image_emb], dim=1)  # (B, bottleneck_dim * 2)

        # --- Regression ---
        output = self.regressor(fused).squeeze(-1)
        return output

