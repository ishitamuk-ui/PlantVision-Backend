# Research Summary: CNN vs Transformer for Plant Disease

## Hypotheses
1. **TransferResNet** will outperform **SimpleCNN** significantly due to pretrained ImageNet features.
2. **VisionTransformer** will achieve comparable or slightly better performance than TransferResNet but may be more computationally expensive during inference.
3. Grad-CAM on CNNs will focus heavily on localized symptomatic lesions, whereas ViTs might capture broader leaf context.

## Experimental Variables
- **Independent Variables**: Model architecture (SimpleCNN, TransferResNet, VisionTransformer).
- **Dependent Variables**: Accuracy, F1-Score, Training time, Inference time.
- **Controlled Variables**: Train/val/test splits, optimizer (Adam), basic data augmentations, random seeds.

## Assumptions
- The training dataset is representative of the test dataset.
- The standard PyTorch ImageNet normalization `[0.485, 0.456, 0.406]` is acceptable for plant leaf images.

## Limitations
- Computational constraints limit the batch size and the size of ViTs we can fine-tune.
- Grad-CAM on ViT might not be as straightforward or perfectly aligned with the attention mechanism.

## Possible Sources of Bias
- Class imbalance in the PlantVillage dataset may bias models toward majority classes if not explicitly handled via weighted sampling (can be toggled in config).

## Next Experiments (Suggested for Reader)
1. **Augmentation Ablation**: Train the models without any augmentation to observe how quickly they overfit.
2. **Reduced Data**: Train with only 10% of the dataset to see if ViT degradation is faster than ResNet.
3. **Class Imbalance**: Toggle `imbalance_strategy: "weighted_sampler"` in config and compare results on minority classes.
