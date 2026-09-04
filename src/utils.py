import os
import random
import numpy as np
import torch
import yaml
import logging

def set_seed(seed: int = 42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path: str) -> dict:
    """Loads a YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_logger(name: str = "PlantVision", level=logging.INFO):
    """Sets up and returns a logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    return logger

def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth'):
    """Saves model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'model_best.pth')
        torch.save(state, best_filepath)

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Loads model checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        
    return model, optimizer, checkpoint.get('epoch', 0), checkpoint.get('best_loss', float('inf'))
