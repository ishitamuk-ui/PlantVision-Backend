import os
import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms
from collections import Counter
import numpy as np
from src.utils import get_logger

logger = get_logger(__name__)

def get_transforms(img_size):
    """Returns training and validation transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def get_dataloaders(config):
    """Loads dataset, applies splits and returns dataloaders."""
    data_dir = config['data']['dataset_dir']
    
    if not os.path.exists(data_dir):
        logger.warning(f"Dataset directory not found: {data_dir}. Please ensure dataset is downloaded.")
        return None, None, None, None

    img_size = config['data']['img_size']
    batch_size = config['data']['batch_size']
    num_workers = config['data']['num_workers']
    
    train_trans, val_trans = get_transforms(img_size)
    
    # Load entire dataset with validation transform initially to get sizes
    full_dataset = datasets.ImageFolder(root=data_dir, transform=val_trans)
    class_names = full_dataset.classes
    logger.info(f"Found {len(class_names)} classes: {class_names}")
    
    total_size = len(full_dataset)
    train_size = int(config['data']['train_split'] * total_size)
    val_size = int(config['data']['val_split'] * total_size)
    test_size = total_size - train_size - val_size
    
    generator = torch.Generator().manual_seed(config['training']['seed'])
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )
    
    # Apply proper transforms
    # We need to create a wrapper dataset to apply different transforms to subsets
    class TransformWrapper(torch.utils.data.Dataset):
        def __init__(self, subset, transform=None):
            self.subset = subset
            self.transform = transform
            
        def __getitem__(self, index):
            x, y = self.subset[index]
            # ImageFolder returns already transformed image (tensor) if transform was passed.
            # We bypass the subset's parent dataset transform by accessing original paths if possible,
            # but simpler approach: load dataset without transform, apply in wrapper.
            return x, y
            
        def __len__(self):
            return len(self.subset)
            
    # Better approach: recreate dataset
    base_dataset_train = datasets.ImageFolder(root=data_dir, transform=train_trans)
    base_dataset_val = datasets.ImageFolder(root=data_dir, transform=val_trans)
    
    train_dataset = torch.utils.data.Subset(base_dataset_train, train_dataset.indices)
    val_dataset = torch.utils.data.Subset(base_dataset_val, val_dataset.indices)
    test_dataset = torch.utils.data.Subset(base_dataset_val, test_dataset.indices)

    # Imbalance handling
    sampler = None
    if config['data'].get('imbalance_strategy') == 'weighted_sampler':
        targets = [base_dataset_train.targets[i] for i in train_dataset.indices]
        class_counts = Counter(targets)
        num_samples = len(targets)
        class_weights = {k: num_samples / v for k, v in class_counts.items()}
        sample_weights = [class_weights[t] for t in targets]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=num_samples, replacement=True)
        shuffle = False # mutually exclusive with sampler
    else:
        shuffle = True
        
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, 
        sampler=sampler, num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, class_names
