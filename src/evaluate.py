import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from src.utils import get_logger

logger = get_logger(__name__)

def evaluate_model(model, test_loader, device, class_names, config, model_name):
    """Evaluates the model on the test set and calculates metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    try:
        # ROC-AUC requires one-hot encoded true labels for multi-class
        if len(class_names) > 2:
            roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
        else:
            roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
        metrics['roc_auc'] = roc_auc
    except Exception as e:
        logger.warning(f"Could not calculate ROC-AUC: {e}")
        metrics['roc_auc'] = None

    logger.info(f"Evaluation Metrics for {model_name}:")
    for k, v in metrics.items():
        if v is not None:
            logger.info(f"{k}: {v:.4f}")
            
    # Save Metrics
    metrics_dir = os.path.join(config['training']['save_dir'], 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    with open(os.path.join(metrics_dir, f'{model_name}_test_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Generate Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    
    figures_dir = os.path.join(config['training']['save_dir'], 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, f'{model_name}_confusion_matrix.png'))
    plt.close()
    
    return metrics
