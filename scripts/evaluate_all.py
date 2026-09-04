import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.utils import load_config, set_seed, get_logger, load_checkpoint
from src.data import get_dataloaders
from src.models import get_model
from src.evaluate import evaluate_model
from src.explainability import run_grad_cam

logger = get_logger("evaluate_all")

def main():
    config_path = os.path.join("configs", "default_config.yaml")
    config = load_config(config_path)
    
    set_seed(config['training']['seed'])
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    
    _, _, test_loader, class_names = get_dataloaders(config)
    if test_loader is None:
        return
        
    num_classes = len(class_names)
    
    for model_name in config['models']['selected']:
        logger.info(f"--- Evaluating {model_name} ---")
        
        model = get_model(model_name, num_classes, config).to(device)
        
        # Load best weights
        checkpoint_dir = os.path.join(config['training']['save_dir'], model_name)
        best_model_path = os.path.join(checkpoint_dir, 'model_best.pth')
        
        if not os.path.exists(best_model_path):
            logger.warning(f"No checkpoint found for {model_name} at {best_model_path}. Skipping.")
            continue
            
        model, _, _, _ = load_checkpoint(best_model_path, model)
        
        # Evaluate
        evaluate_model(model, test_loader, device, class_names, config, model_name)
        
        # Run explainability on a single batch
        logger.info(f"Running Grad-CAM for {model_name}...")
        inputs, labels = next(iter(test_loader))
        
        # Pick the first image in the batch
        image_tensor = inputs[0].to(device)
        label = labels[0].item()
        
        # Reconstruct image for visualization
        # Image is normalized, we need to denormalize for visualization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        unnorm_image = inputs[0] * std + mean
        unnorm_image = torch.clamp(unnorm_image, 0, 1)
        original_image_np = unnorm_image.permute(1, 2, 0).numpy()
        
        run_grad_cam(model, model_name, image_tensor, label, original_image_np, config)
        
if __name__ == "__main__":
    main()
