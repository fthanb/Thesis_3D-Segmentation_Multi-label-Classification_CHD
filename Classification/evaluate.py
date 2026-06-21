import torch
import numpy as np
import warnings
import os
import sys
import logging
from datetime import datetime
from contextlib import redirect_stdout
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, classification_report, average_precision_score

warnings.filterwarnings('ignore')

from dataset import CHDMultiLabelDataset, get_val_test_transforms
from model import build_hybrid_model

def setup_eval_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_filename = datetime.now().strftime("eval_%Y%m%d_%H%M%S.log")
    log_filepath = os.path.join(log_dir, log_filename)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s", 
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def evaluate_chd_model(model_weights_path, test_csv_path, image_dir, nnunet_model_dir, logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_transforms = get_val_test_transforms()
    test_dataset = CHDMultiLabelDataset(test_csv_path, image_dir, transform=test_transforms)
 
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    class_names = test_dataset.label_columns
    num_classes = len(class_names)
    
    with open(os.devnull, 'w') as f, redirect_stdout(f):
        model = build_hybrid_model(nnunet_model_dir, fold_index=0, num_chd_classes=num_classes)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    all_targets = []
    all_predictions = []
    all_probabilities = []
    class_thresholds = torch.tensor([
        0.50,  # ASD 
        0.50,  # VSD
        0.30,  # ToF 
        0.30,  # TGA 
        0.30,  # DORV 
        0.40,  # CAT 
        0.30,  # CA
        0.20,  # AAH
        0.20,  # DAA
        0.30,  # IAA
        0.20,  # PA 
        0.40,  # APVC 
        0.25,  # DSVC 
        0.25,  # PDA 
        0.25   # PAS 
    ]).to(device)
    
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            with torch.cuda.amp.autocast():
                logits = model(images)
            
            probs = torch.sigmoid(logits)
            
            preds = (probs > class_thresholds).float()
            
            all_probabilities.append(probs.cpu().numpy())
            all_predictions.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())
            
    y_true = np.vstack(all_targets)
    y_pred = np.vstack(all_predictions)
    y_prob = np.vstack(all_probabilities)
    
    logger.info("\nGNN metrics")
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    logger.info(f"Macro F1-Score : {macro_f1:.4f}")
    logger.info(f"Micro F1-Score : {micro_f1:.4f}")
    
    try:
        roc_auc = roc_auc_score(y_true, y_prob, average='macro')
        logger.info(f"ROC-AUC Score  : {roc_auc:.4f}")
    except ValueError:
        logger.info("ROC-AUC Score  : Undefined")
        
    try:
        map_score = average_precision_score(y_true, y_prob, average='macro')
        logger.info(f"mAP Score      : {map_score:.4f}")
    except ValueError:
        logger.info("mAP Score      : Undefined")
        
    logger.info("\nMetric per class")
    
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    logger.info(report)

if __name__ == "__main__":
    LOG_DIR = "/home/s2516118/skripsi/logs"
    logger = setup_eval_logger(LOG_DIR)
    
    CSV_DIR = "/home/s2516118/skripsi/info"
    IMAGE_DIR = "/home/s2516118/skripsi/U-Mamba/data/nnUNet_raw/Dataset001_ImageCHD/imagesTr"
    NNUNET_MODEL_DIR = "/home/s2516118/skripsi/U-Mamba/data/nnUNet_results/Dataset001_ImageCHD/nnUNetTrainerUMambaBot__nnUNetPlans__3d_fullres"
    
    TEST_CSV_PATH = f"{CSV_DIR}/test_data.csv"
    
    models_to_evaluate = ["best_model.pth", "final_model.pth"]
    
    for model_weights in models_to_evaluate:
        if os.path.exists(model_weights):
            logger.info(f"Evaluating {model_weights}")
            
            evaluate_chd_model(model_weights, TEST_CSV_PATH, IMAGE_DIR, NNUNET_MODEL_DIR, logger)
        else:
            logger.info(f"{model_weights} not found.")