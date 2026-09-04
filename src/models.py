import torch
import torch.nn as nn
import torchvision.models as models

class SimpleCNN(nn.Module):
    """A simple CNN baseline trained from scratch."""
    def __init__(self, num_classes, img_size=224, dropout_rate=0.3):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 112
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 56
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 28
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 14
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(256 * (img_size // 16) * (img_size // 16), 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class TransferResNet(nn.Module):
    """Pretrained ResNet model for transfer learning."""
    def __init__(self, num_classes, fine_tune_layers=2):
        super(TransferResNet, self).__init__()
        
        # Load pretrained ResNet18
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Freeze layers
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Unfreeze the last `fine_tune_layers` blocks
        if fine_tune_layers > 0:
            layers_to_unfreeze = [self.model.layer4, self.model.layer3, self.model.layer2, self.model.layer1][-fine_tune_layers:]
            for layer in layers_to_unfreeze:
                for param in layer.parameters():
                    param.requires_grad = True
                    
        # Replace the final fully connected layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

class VisionTransformer(nn.Module):
    """Pretrained Vision Transformer (ViT)."""
    def __init__(self, num_classes, model_name="vit_b_16"):
        super(VisionTransformer, self).__init__()
        
        if model_name == "vit_b_16":
            self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported ViT model: {model_name}")
            
        # Freeze majority of the layers
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Replace the heads
        num_ftrs = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(num_ftrs, num_classes)
        
        # Unfreeze the last encoder block and the head
        for param in self.model.encoder.layers[-1].parameters():
            param.requires_grad = True
        for param in self.model.heads.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)

def get_model(model_name, num_classes, config):
    """Factory function to get the requested model."""
    if model_name == 'SimpleCNN':
        dropout_rate = config['models']['SimpleCNN'].get('dropout_rate', 0.3)
        img_size = config['data'].get('img_size', 224)
        return SimpleCNN(num_classes, img_size, dropout_rate)
        
    elif model_name == 'TransferResNet':
        ft_layers = config['models']['TransferResNet'].get('fine_tune_layers', 2)
        return TransferResNet(num_classes, fine_tune_layers=ft_layers)
        
    elif model_name == 'VisionTransformer':
        vit_name = config['models']['VisionTransformer'].get('model_name', 'vit_b_16')
        return VisionTransformer(num_classes, model_name=vit_name)
        
    else:
        raise ValueError(f"Model {model_name} not recognized.")
