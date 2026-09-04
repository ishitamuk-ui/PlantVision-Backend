import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
from src.utils import get_logger, save_checkpoint

logger = get_logger(__name__)

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_model(model, train_loader, val_loader, criterion, optimizer, device, config, model_name):
    epochs = config['training']['epochs']
    patience = config['training']['patience']
    save_dir = os.path.join(config['training']['save_dir'], model_name)
    os.makedirs(save_dir, exist_ok=True)
    
    early_stopping = EarlyStopping(patience=patience)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'epoch_time': []}
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training Phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})
            
        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total
        
        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({'loss': loss.item(), 'acc': val_correct/val_total})
                
        epoch_val_loss = val_running_loss / val_total
        epoch_val_acc = val_correct / val_total
        epoch_time = time.time() - start_time
        
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['epoch_time'].append(epoch_time)
        
        logger.info(f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} - "
                    f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f} - "
                    f"Time: {epoch_time:.2f}s")
        
        # Checkpoint
        is_best = epoch_val_loss < best_val_loss
        if is_best:
            best_val_loss = epoch_val_loss
            
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_val_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best, save_dir, filename=f'checkpoint_epoch_{epoch+1}.pth')
        
        # Early Stopping
        early_stopping(epoch_val_loss)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered.")
            break
            
    # Save history
    metrics_dir = os.path.join(config['training']['save_dir'], 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    pd.DataFrame(history).to_csv(os.path.join(metrics_dir, f'{model_name}_training_history.csv'), index=False)
    
    return history
