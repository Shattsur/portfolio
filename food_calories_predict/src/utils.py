# utils.py
import os
import csv
import random
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import re
from torch.cuda.amp import autocast, GradScaler

# ===== Сидирование =====
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ===== Разморозка параметров =====
def set_requires_grad(model, unfreeze_layers, model_name=""):
    """
    unfreeze_layers:
      - None -> ничего не трогаем
      - str (regex) -> используем re.search
      - iterable of substrings -> any(sub in name)
    """
    if not unfreeze_layers:
        print(f"🔒 {model_name}: все слои заморожены (unfreeze_layers=None)")
        return
    
    print(f"\n🎯 {model_name}: начинаем разморозку...")
    
    total_params = 0
    unfrozen_params = 0
    unfrozen_layers = set()
    
    if isinstance(unfreeze_layers, str):
        pattern = re.compile(unfreeze_layers)
        
        for n, p in model.named_parameters():
            total_params += 1
            if pattern.search(n):
                p.requires_grad_(True)
                unfrozen_params += 1
                # Извлекаем имя слоя для группировки
                layer_name = extract_layer_name(n)
                unfrozen_layers.add(layer_name)
            else:
                p.requires_grad_(False)
    else:
        # предполагаем iterable of substrings
        for n, p in model.named_parameters():
            total_params += 1
            if any(sub in n for sub in unfreeze_layers):
                p.requires_grad_(True)
                unfrozen_params += 1
                layer_name = extract_layer_name(n)
                unfrozen_layers.add(layer_name)
            else:
                p.requires_grad_(False)
    
    # ТОЛЬКО СТАТИСТИКА, без подробного вывода
    print(f"📊 {model_name} - ИТОГО:")
    print(f"   • Всего параметров: {total_params}")
    print(f"   • Разморожено: {unfrozen_params} ({unfrozen_params/total_params*100:.1f}%)")
    print(f"   • Заморожено: {total_params - unfrozen_params}")
    
    if unfrozen_layers:
        print(f"   • Размороженные слои: {sorted(unfrozen_layers)}")
    
    return unfrozen_params, total_params

def extract_layer_name(full_name):
    """Извлекает понятное имя слоя из полного имени параметра"""
    if 'encoder.layer.' in full_name:
        match = re.search(r'encoder\.layer\.\d+', full_name)
        if match:
            return match.group(0)
    elif 'blocks.' in full_name:
        match = re.search(r'blocks\.\d+', full_name)
        if match:
            return match.group(0)
    elif 'layernorm' in full_name.lower():
        return 'layernorm'
    elif 'norm' in full_name.lower():
        return 'norm'
    elif 'register_tokens' in full_name:
        return 'register_tokens'
    elif 'embeddings' in full_name:
        return 'embeddings'
    elif 'pooler' in full_name:
        return 'pooler'
    
    return full_name.split('.')[0]

# ===== Функции потерь =====
class AdaptiveHuberLoss(nn.Module):
    def __init__(self, delta=1.0, adaptive=True):
        super().__init__()
        self.adaptive = adaptive
        self.delta = nn.Parameter(torch.tensor(delta)) if adaptive else delta

    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        if self.adaptive and diff.numel() > 1:
            with torch.no_grad():
                self.delta.data = torch.quantile(diff.detach(), 0.8).clamp(0.1, 10.0)
        delta = self.delta if self.adaptive else torch.tensor(self.delta, device=pred.device)
        loss = torch.where(diff < delta, 0.5 * diff ** 2, delta * (diff - 0.5 * delta))
        return loss.mean()

class WeightedMSELoss(nn.Module):
    def __init__(self, target_mean, target_std):
        super().__init__()
        self.target_mean = target_mean
        self.target_std = target_std

    def forward(self, pred_norm, target_norm, target_original=None):
        if target_original is None:    
            target_original = target_norm * self.target_std + self.target_mean

        weights = torch.where(
            target_original > 750.0, 10.0,  # Максимальный штраф для самых высоких
            torch.where(
                target_original > 500.0, 7.0,  # Увеличенный штраф для > 500
                torch.where(
                    target_original > 300.0, 5.0, # Увеличенный штраф для > 300
                    1.0 # Стандартный штраф для <= 300
                )
            )
        )
        return (weights * (pred_norm - target_norm) ** 2).mean()

class CombinedLoss(nn.Module):
    def __init__(self, mse_weight=0.9, huber_weight=0.2):
        super().__init__()
        self.mse_weight = mse_weight
        self.huber_weight = huber_weight
        self.mse_loss = nn.MSELoss()
        self.huber_loss = AdaptiveHuberLoss()

    def forward(self, pred, target, target_original=None):
        return self.mse_weight * self.mse_loss(pred, target) + self.huber_weight * self.huber_loss(pred, target)

# ===== Валидация С ЧИСЛОВЫМИ ПРИЗНАКАМИ =====
def validate_regression(model, val_loader, device, criterion=None, target_mean=None, target_std=None, config=None):
    """
    Валидация: loss считается в нормализованном пространстве (как в train),
    MAE возвращается в оригинальной шкале. Собираем статистику по диапазонам.
    Поддержка числовых признаков.
    """
    model.eval()
    total_loss = 0.0
    total_abs_error = 0.0
    total_samples = 0
    calorie_ranges = [(0,200),(200,400),(400,600),(600,800),(800,2000)]
    range_errors = {f"{l}-{h}": [] for l,h in calorie_ranges}

    if criterion is None:
        criterion = nn.HuberLoss(delta=1.0)

    # Используем значения из конфига, если не переданы явно
    if target_mean is None:
        target_mean = config.TARGET_MEAN
    if target_std is None:
        target_std = config.TARGET_STD

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["image"].to(device)
            labels = batch["label"].to(device)  # original scale
            
            # Получаем числовые признаки если они есть
            numeric_features = None
            if config and getattr(config, 'USE_NUMERIC_FEATURES', False) and 'numeric' in batch:
                numeric_features = batch["numeric"].to(device)

            # model возвращает normalized predictions
            if numeric_features is not None:
                preds_norm = model(input_ids, attention_mask, images, numeric_features)
            else:
                preds_norm = model(input_ids, attention_mask, images)

            # считаем loss в нормализованном пространстве
            labels_norm = (labels - target_mean) / target_std
            loss = criterion(preds_norm, labels_norm)

            # переводим предсказания в оригинальную шкалу для MAE/отчёта
            preds_orig = preds_norm * target_std + target_mean

            bsz = labels.size(0)
            total_loss += float(loss.item()) * bsz
            total_abs_error += float(torch.abs(preds_orig - labels).sum().item())
            total_samples += bsz

            # диапазоны (по оригинальным меткам)
            for i in range(bsz):
                lab = float(labels[i].item())
                pred_val = float(preds_orig[i].item())
                err = abs(pred_val - lab)
                for low, high in calorie_ranges:
                    if low <= lab < high:
                        range_errors[f"{low}-{high}"].append(err)
                        break

    if total_samples == 0:
        return float("inf"), float("inf"), {}

    avg_loss = total_loss / total_samples
    avg_mae = total_abs_error / total_samples

    range_stats = {}
    for rng, errs in range_errors.items():
        if errs:
            arr = np.array(errs, dtype=float)
            range_stats[rng] = {'mae': float(arr.mean()), 'samples': int(len(arr)), 'std': float(arr.std())}

    return avg_loss, avg_mae, range_stats

# ===== ПЛОТИНГ =====
def plot_training_analysis(train_maes, val_maes, grad_norms, lr_history, config):
    if not train_maes: 
        print("⚠️ Нет данных для графиков")
        return
    
    epochs = range(1, len(train_maes) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # MAE
    axes[0, 0].plot(epochs, train_maes, 'b-o', label="Train MAE")
    axes[0, 0].plot(epochs, val_maes, 'r-s', label="Val MAE")
    axes[0, 0].set_title("MAE Train vs Val")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Grad Norm
    axes[0, 1].plot(epochs, grad_norms, 'g-', label="Grad Norm")
    axes[0, 1].set_title("Gradient Norms")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # LR
    axes[1, 0].plot(epochs, lr_history, 'purple', label="LR")
    axes[1, 0].set_title("Learning Rate")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')

    # Overfitting gap
    diff = [val_maes[i] - train_maes[i] for i in range(len(train_maes))]
    axes[1, 1].plot(epochs, diff, 'orange', label="Val - Train")
    axes[1, 1].axhline(0, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_title("Overfitting Gap")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(config.BASE_DIR / "training_analysis.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("📈 Графики обучения сохранены.")

# ===== ТРЕНИРОВОЧНЫЙ ЦИКЛ С ЧИСЛОВЫМИ ПРИЗНАКАМИ =====
def train_regression(config, train_loader, val_loader):
    seed_everything(config.SEED)
    device = config.DEVICE

    from model import MultimodalModel
    model = MultimodalModel(config).to(device)
    print(f"✅ Model initialized on {device}")
    
    # Информация о числовых признаках
    if getattr(config, 'USE_NUMERIC_FEATURES', False):
        print(f"📊 Using numeric features: n_ingredients")
    
    print("ДИАГНОСТИКА РАЗМОРОЗКИ")
    
    # Разморозка с компактной диагностикой
    text_unfrozen, text_total = set_requires_grad(
        model.text_model, 
        getattr(config, "TEXT_UNFREEZE", None),
        "BGE Text Model"
    )
    
    image_unfrozen, image_total = set_requires_grad(
        model.image_model, 
        getattr(config, "IMAGE_UNFREEZE", None), 
        "DINOv2 Image Model"
    )
    
    # Сводная статистика
    print(f"\n📈 СВОДНАЯ СТАТИСТИКА:")
    print(f"📝 Текстовая модель: {text_unfrozen}/{text_total} разморожено ({text_unfrozen/text_total*100:.1f}%)")
    print(f"🖼️  Изображенческая модель: {image_unfrozen}/{image_total} разморожено ({image_unfrozen/image_total*100:.1f}%)")
    
    # Итоговый отчет по всей модели
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n🎯 ИТОГО ПО МОДЕЛИ:")
    print(f"   • Всего параметров: {total_params:,}")
    print(f"   • Обучаемых: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    print(f"   • Замороженных: {total_params - trainable_params:,}")
    
    # Продолжаем обычный тренировочный цикл...
    def get_param_groups(model, config):
        no_decay = ["bias", "LayerNorm.weight"]
        param_groups = []

        # TEXT - только размороженные параметры
        text_trainable = [p for n, p in model.text_model.named_parameters() if p.requires_grad]
        if text_trainable:
            # Разделяем на decay и no_decay
            decay_params = [p for p in text_trainable if not any(nd in n for nd in no_decay for n, param in model.text_model.named_parameters() if param is p)]
            no_decay_params = [p for p in text_trainable if any(nd in n for nd in no_decay for n, param in model.text_model.named_parameters() if param is p)]
            
            if decay_params:
                param_groups.append({
                    "params": decay_params,
                    "lr": getattr(config, "LR_TEXT", 1e-5),
                    "weight_decay": getattr(config, "WEIGHT_DECAY", 1e-3)
                })
            if no_decay_params:
                param_groups.append({
                    "params": no_decay_params,
                    "lr": getattr(config, "LR_TEXT", 1e-5),
                    "weight_decay": 0.0
                })

        # IMAGE - только размороженные параметры
        image_trainable = [p for n, p in model.image_model.named_parameters() if p.requires_grad]
        if image_trainable:
            decay_params = [p for p in image_trainable if not any(nd in n for nd in no_decay for n, param in model.image_model.named_parameters() if param is p)]
            no_decay_params = [p for p in image_trainable if any(nd in n for nd in no_decay for n, param in model.image_model.named_parameters() if param is p)]
            
            if decay_params:
                param_groups.append({
                    "params": decay_params,
                    "lr": getattr(config, "LR_IMAGE", 1e-5),
                    "weight_decay": getattr(config, "WEIGHT_DECAY", 1e-3)
                })
            if no_decay_params:
                param_groups.append({
                    "params": no_decay_params,
                    "lr": getattr(config, "LR_IMAGE", 1e-5),
                    "weight_decay": 0.0
                })

        # HEAD (всегда разморожены)
        head_params = (
            list(model.text_bottleneck.parameters()) + 
            list(model.image_bottleneck.parameters()) + 
            list(model.regressor.parameters())
        )
        
        # Добавляем numeric_bottleneck если он есть
        if hasattr(model, 'numeric_bottleneck') and model.numeric_bottleneck is not None:
            head_params += list(model.numeric_bottleneck.parameters())
            
        param_groups.append({
            "params": head_params,
            "lr": getattr(config, "LR_HEAD", 1e-4),
            "weight_decay": getattr(config, "WEIGHT_DECAY", 1e-3)
        })

        return param_groups

    optimizer = torch.optim.AdamW(get_param_groups(model, config))

    # Используем ReduceLROnPlateau для отслеживания val_mae
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        min_lr=1e-8       
    )

    # Loss - используем вычисленные значения из конфига
    loss_type = getattr(config, "LOSS_TYPE", "huber")
    if loss_type == "huber":
        criterion = AdaptiveHuberLoss()
    elif loss_type == "weighted_mse":
        criterion = WeightedMSELoss(config.TARGET_MEAN, config.TARGET_STD)
    elif loss_type == "combined":
        criterion = CombinedLoss()
    else:
        criterion = nn.MSELoss()
    print(f"🎯 Loss type: {loss_type}")

    scaler = GradScaler()
    best_val_mae = float("inf")
    patience, wait = getattr(config, "PATIENCE", 8), 0
    
    # Используем вычисленные значения из конфига
    target_mean = config.TARGET_MEAN
    target_std = config.TARGET_STD
    
    print(f"📊 Используется статистика: mean={target_mean:.2f}, std={target_std:.2f}")

    log_path = config.BASE_DIR / "enhanced_train_log.csv"
    os.makedirs(config.BASE_DIR, exist_ok=True)
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_mae", "val_mae", "val_loss", "lr", "grad_norm", "nan_batches"])

    train_maes, val_maes, grad_norms, lr_history = [], [], [], []

    for epoch in range(config.EPOCHS):
        model.train()
        total_abs_error, total_samples, grad_sum, nan_batches = 0.0, 0, 0.0, 0
        processed_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["image"].to(device, dtype=torch.float) 

            # Получаем числовые признаки если они есть
            numeric_features = None
            if getattr(config, 'USE_NUMERIC_FEATURES', False) and 'numeric' in batch:
                numeric_features = batch["numeric"].to(device)

            use_mixup = bool(batch.get("use_mixup", False))
            if use_mixup:
                labels_a = batch["labels_a"].to(device)
                labels_b = batch["labels_b"].to(device)
                lam = float(batch.get("lam", 1.0))
                labels_a_norm = (labels_a - target_mean) / target_std
                labels_b_norm = (labels_b - target_mean) / target_std
            else:
                labels = batch["label"].to(device)
                labels_norm = (labels - target_mean) / target_std

            optimizer.zero_grad()

            with autocast():
                # Передаем numeric_features в модель
                if numeric_features is not None:
                    preds_norm = model(input_ids, attention_mask, images, numeric_features)
                else:
                    preds_norm = model(input_ids, attention_mask, images)
                    
                if torch.isnan(preds_norm).any():
                    nan_batches += 1
                    continue
                if use_mixup:
                    loss = lam * criterion(preds_norm, labels_a_norm) + (1-lam) * criterion(preds_norm, labels_b_norm)
                else:
                    loss = criterion(preds_norm, labels_norm)
                if torch.isnan(loss):
                    nan_batches += 1
                    continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), getattr(config, "GRAD_CLIP_NORM", 1.0))
            grad_sum += float(grad_norm)
            scaler.step(optimizer)
            scaler.update()

            preds_orig = preds_norm * target_std + target_mean
            if use_mixup:
                total_abs_error += (lam * torch.abs(preds_orig - labels_a).sum() + (1-lam) * torch.abs(preds_orig - labels_b).sum()).item()
                total_samples += labels_a.size(0)
            else:
                total_abs_error += torch.abs(preds_orig - labels).sum().item()
                total_samples += labels.size(0)
            processed_batches += 1

        # === ВАЛИДАЦИЯ в конце эпохи ===
        train_mae = total_abs_error / max(total_samples, 1)
        avg_grad_norm = (grad_sum / processed_batches) if processed_batches > 0 else 0.0
        val_loss, val_mae, _ = validate_regression(model, val_loader, device, criterion, target_mean, target_std, config)
        
        # Обновляем scheduler ПОСЛЕ валидации
        scheduler.step(val_mae)
        
        # Получаем текущий learning rate из optimizer'а
        current_lr = optimizer.param_groups[0]['lr']

        # Логирование
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch+1, train_mae, val_mae, val_loss, current_lr, avg_grad_norm, nan_batches])
        print(f"\nEpoch {epoch+1}: Train MAE={train_mae:.1f}, Val MAE={val_mae:.1f}, Grad={avg_grad_norm:.3f}, NaN={nan_batches}")

        train_maes.append(train_mae)
        val_maes.append(val_mae)
        grad_norms.append(avg_grad_norm)
        lr_history.append(current_lr)

        # Early stopping
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            wait = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_mae': best_val_mae,
                'epoch': epoch,
                'config': getattr(config, "__dict__", None)
            }, config.SAVE_PATH)
            print(f"💾 Saved best model (val_mae={best_val_mae:.1f})")
        else:
            wait += 1
            if wait >= patience:
                print("⏹ Early stopping.")
                break

    print(f"\n✅ Training finished. Best val_mae: {best_val_mae:.1f}")

    try:
        plot_training_analysis(train_maes, val_maes, grad_norms, lr_history, config)
    except Exception as e:
        print(f"⚠️ Plotting failed: {e}")

    return model