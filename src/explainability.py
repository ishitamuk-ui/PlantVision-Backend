import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torchvision.transforms.functional as TF
from src.utils import get_logger

logger = get_logger(__name__)

def run_grad_cam(model, model_name, image_tensor, target_class, original_image_np, config):
    """
    Runs Grad-CAM on a specific model for a given image.
    original_image_np should be a normalized float32 numpy array (H, W, C) in range [0, 1].
    """
    # Determine the target layer based on the model architecture
    target_layers = []
    
    if model_name == 'SimpleCNN':
        # Last conv layer before MaxPool
        target_layers = [model.features[-3]] 
    elif model_name == 'TransferResNet':
        # Last layer of ResNet
        target_layers = [model.model.layer4[-1]]
    elif model_name == 'VisionTransformer':
        # Grad-CAM on ViT requires careful reshaping, pytorch-grad-cam handles it via specific reshape_transform
        def reshape_transform(tensor, height=14, width=14):
            # tensor shape is [batch, num_tokens, hidden_dim]
            result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
            # Bring the channels to the first dimension
            result = result.transpose(2, 3).transpose(1, 2)
            return result
        
        target_layers = [model.model.encoder.layers[-1].ln_1]
        
    else:
        logger.warning(f"Grad-CAM not implemented for {model_name}")
        return
        
    try:
        if model_name == 'VisionTransformer':
            cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
        else:
            cam = GradCAM(model=model, target_layers=target_layers)
            
        targets = [ClassifierOutputTarget(target_class)]
        
        # Generate heatmap
        grayscale_cam = cam(input_tensor=image_tensor.unsqueeze(0), targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        # Overlay heatmap on original image
        visualization = show_cam_on_image(original_image_np, grayscale_cam, use_rgb=True)
        
        # Save visualization
        figures_dir = os.path.join(config['training']['save_dir'], 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(original_image_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(visualization)
        axes[1].set_title(f'Grad-CAM (Class {target_class})')
        axes[1].axis('off')
        
        save_path = os.path.join(figures_dir, f'{model_name}_gradcam_class_{target_class}.png')
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"Grad-CAM visualization saved to {save_path}")
        
    except Exception as e:
        logger.error(f"Error running Grad-CAM for {model_name}: {e}")
