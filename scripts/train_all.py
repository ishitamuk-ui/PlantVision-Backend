import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from src.utils import load_config, set_seed, get_logger
from src.data import get_dataloaders
from src.models import get_model
from src.train import train_model

logger = get_logger("train_all")

def main():
    config_path = os.path.join("configs", "default_config.yaml")
    config = load_config(config_path)
    
    set_seed(config['training']['seed'])
    
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    train_loader, val_loader, _, class_names = get_dataloaders(config)
    if train_loader is None:
        logger.error("Failed to load data. Exiting.")
        return
        
    num_classes = len(class_names)
    
    for model_name in config['models']['selected']:
        logger.info(f"--- Training {model_name} ---")
        model = get_model(model_name, num_classes, config).to(device)
        
        criterion = nn.CrossEntropyLoss()
        
        # Get model specific learning rate
        lr = config['models'][model_name].get('learning_rate', config['training']['learning_rate'])
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=float(config['training']['weight_decay']))
        
        history = train_model(model, train_loader, val_loader, criterion, optimizer, device, config, model_name)
        
        # Plot training curves
        figures_dir = os.path.join(config['training']['save_dir'], 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title(f'{model_name} Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Acc')
        plt.plot(history['val_acc'], label='Val Acc')
        plt.title(f'{model_name} Accuracy')
        plt.legend()
        
        plt.savefig(os.path.join(figures_dir, f'{model_name}_learning_curves.png'))
        plt.close()
        
        logger.info(f"Finished training {model_name}\n")

if __name__ == "__main__":
    main()
