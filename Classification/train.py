import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import logging
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from contextlib import redirect_stdout
from datetime import datetime

from dataset import CHDMultiLabelDataset, get_standard_transforms, get_heavy_transforms, get_val_test_transforms
from model import build_hybrid_model

def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    
    log_filename = datetime.now().strftime("training_%Y%m%d_%H%M%S.log")
    log_filepath = os.path.join(log_dir, log_filename)

    class SafeFileHandler(logging.Handler):
        def __init__(self, filepath):
            super().__init__()
            self.filepath = filepath

        def emit(self, record):
            try:
                msg = self.format(record)
                with open(self.filepath, 'a') as f:
                    f.write(msg + '\n')
            except Exception:
                pass

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
 
    if logger.hasHandlers():
        logger.handlers.clear()
        
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    
    file_handler = SafeFileHandler(log_filepath)
    file_handler.setFormatter(formatter)
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger

class MultiLabelFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super(MultiLabelFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        
        if self.label_smoothing > 0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - targets)
            focal_weight = focal_weight * alpha_t
            
        loss = focal_weight * bce_loss
        return loss.mean()    

# TRAINING & VALIDATION
def train_chd_topology_model(model, train_loader, val_loader, pos_weight, logger, num_epochs, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    trainable_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_parameters, lr=learning_rate, weight_decay=1e-3)
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=learning_rate, 
        epochs=num_epochs, 
        steps_per_epoch=steps_per_epoch,
        pct_start=0.2, 
        div_factor=10.0, 
        final_div_factor=100.0
    )
    criterion = MultiLabelFocalLoss(alpha=pos_weight.to(device), gamma=2.0)
    scaler = torch.cuda.amp.GradScaler()
    
    best_val_loss = float('inf')
    
    logger.info("Start Training")
    for epoch in range(num_epochs):

        model.train()
        running_train_loss = 0.0
        
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, targets)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scale_before = scaler.get_scale()
            scaler.update()
            scale_after = scaler.get_scale()
            if scale_before <= scale_after:
                scheduler.step()
            
            running_train_loss += loss.item()
            
            current_lr = optimizer.param_groups[0]['lr']
            
        avg_train_loss = running_train_loss / len(train_loader)
        
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for val_images, val_targets in val_loader:
                val_images, val_targets = val_images.to(device), val_targets.to(device)
                
                with torch.cuda.amp.autocast():
                    val_logits = model(val_images)
                    v_loss = criterion(val_logits, val_targets)
                    running_val_loss += v_loss.item()
                    
        avg_val_loss = running_val_loss / len(val_loader)

        logger.info(f"Epoch [{epoch+1}] -> Train Loss: {avg_train_loss:.3f} | Val Loss: {avg_val_loss:.3f} | LR: {current_lr:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            logger.info(f"Saving best model (Loss: {best_val_loss:.3f})")

    torch.save(model.state_dict(), "final_model.pth")  
    logger.info("Completed")

# MAIN
if __name__ == "__main__":
    LOG_DIR = "/home/s2516118/skripsi/logs"
    logger = setup_logger(LOG_DIR)
    
    CSV_DIR = "/home/s2516118/skripsi/info"
    IMAGE_DIR = "/home/s2516118/skripsi/U-Mamba/data/nnUNet_raw/Dataset001_ImageCHD/imagesTr"
    NNUNET_MODEL_DIR = "/home/s2516118/skripsi/U-Mamba/data/nnUNet_results/Dataset001_ImageCHD/nnUNetTrainerUMambaBot__nnUNetPlans__3d_fullres"
    
    # Hyperparameters
    NUM_EPOCHS = 50
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 15
    BEST_WEIGHTS_PATH = "best_model.pth"

    logger.info("Data Loaders")
    train_dataset = CHDMultiLabelDataset(f"{CSV_DIR}/train_data.csv", IMAGE_DIR, transform=get_standard_transforms(), heavy_transform=get_heavy_transforms())
  
    logger.info("WeightedRandomSampler for Rare Classes")
    train_labels = train_dataset.patient_data.iloc[:, 1:].values.astype(float)
    
    class_counts = train_labels.sum(axis=0)
    class_weights_inv = 1.0 / (class_counts + 1e-5)
    
    sample_weights = (train_labels * class_weights_inv).max(axis=1)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights) * 2, 
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=sampler,     
        num_workers=0, 
        pin_memory=True
    )
    
    val_dataset = CHDMultiLabelDataset(f"{CSV_DIR}/val_data.csv", IMAGE_DIR, transform=get_val_test_transforms())
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    logger.info("Positive weights for imbalanced data")
    labels = train_dataset.patient_data.iloc[:, 1:].values.astype(float)
    num_positives = labels.sum(axis=0)
    num_negatives = len(labels) - num_positives
    
    pos_weight_np = num_negatives / (num_positives + 1e-5) 
    
    pos_weight_tensor = torch.clamp(torch.tensor(pos_weight_np, dtype=torch.float32), max=25.0)

    logger.info("Hybrid Model Architecture")
    with open(os.devnull, 'w') as f, redirect_stdout(f):
        model = build_hybrid_model(NNUNET_MODEL_DIR, fold_index=0, num_chd_classes=NUM_CLASSES)

    train_chd_topology_model(
        model, 
        train_loader, 
        val_loader, 
        pos_weight=pos_weight_tensor,
        logger=logger,
        num_epochs=NUM_EPOCHS, 
        learning_rate=LEARNING_RATE
    )