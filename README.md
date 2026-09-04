# PlantVision: Comparative Study of CNN and Transformer Models for Plant Disease Detection

## Project Overview
This project builds an end-to-end computer vision pipeline to classify plant leaf images into disease or healthy categories. It serves as a comparative study between traditional convolutional neural networks (CNNs) and modern Vision Transformers (ViTs).

## Motivation
With the growing adoption of Transformer architectures in computer vision, it is essential to understand their trade-offs compared to CNNs in specialized domains like agriculture. This project explores how lightweight ViTs compare to standard CNNs (trained from scratch) and ResNet architectures (via transfer learning) regarding accuracy, computational requirements, and explainability.

## Research Question
**"How do CNN, transfer-learning, and Transformer-based architectures compare for plant disease classification under the same experimental setting?"**

## Dataset
We recommend the **PlantVillage** dataset, which contains healthy and diseased leaf images across various crop species.
*Note: The dataset is not included in this repository. Please download it and place it in the `data/PlantVillage` directory.*

## Methodology
The pipeline involves:
1. **Data Preprocessing & Augmentation**: Resizing to 224x224, random flips, rotations, and color jittering to prevent overfitting.
2. **Model Training**: Standardizing the training loop with early stopping, learning rate scheduling (if applicable), and same random seeds for fair comparison.
3. **Evaluation**: Computing standard classification metrics (Accuracy, Precision, Recall, F1-score, ROC-AUC) on a hold-out test set.
4. **Explainability**: Applying Grad-CAM to visualize regions of interest influencing model predictions.

## Model Architectures
1. **SimpleCNN**: A lightweight 4-block custom CNN trained from scratch.
2. **TransferResNet**: A pretrained ResNet-18 model fine-tuned on the dataset.
3. **VisionTransformer**: A pretrained ViT (vit_b_16) adapted for the dataset.

## Experimental Setup
- **Framework**: PyTorch
- **Hyperparameters**: Defined in `configs/default_config.yaml`
- **Seeds**: Fixed random seed (42) for reproducibility.

## Results (Placeholders)
*(Results will be populated after running experiments)*

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| SimpleCNN | TBD | TBD | TBD | TBD | TBD |
| TransferResNet | TBD | TBD | TBD | TBD | TBD |
| VisionTransformer | TBD | TBD | TBD | TBD | TBD |

## Visualizations and Explainability
*(Insert screenshots of loss curves, confusion matrices, and Grad-CAM outputs here)*

## Findings
- **TBD**: Analysis of which model generalized better.
- **TBD**: Analysis of inference time vs performance.
- **TBD**: Discussion of what features the CNN vs Transformer focused on (based on Grad-CAM/attention maps).

## Limitations
- The PlantVillage dataset features leaves in controlled backgrounds, which may not translate well to real-world field images (domain shift).
- ViT models typically require larger datasets to train from scratch, hence the reliance on transfer learning.

## Future Work
- Experiment with field-condition datasets.
- Perform augmentation ablation studies.
- Investigate handling class imbalance more robustly.

## Reproduction Instructions

### 1. Environment Setup
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Dataset Preparation
1. Download the PlantVillage dataset (e.g., from Kaggle).
2. Extract the dataset into `data/PlantVillage/`.
   Ensure the structure looks like:
   ```
   data/PlantVillage/
   ├── Apple___Apple_scab/
   ├── Apple___Black_rot/
   ...
   ```

### 3. Training Models
You can configure hyper-parameters in `configs/default_config.yaml`. To train all models sequentially:
```bash
python scripts/train_all.py
```

### 4. Evaluating Models
To evaluate the best checkpoints and generate confusion matrices and Grad-CAM visualizations:
```bash
python scripts/evaluate_all.py
```

### 5. Jupyter Notebooks
Explore data and run interactive experiments:
```bash
jupyter notebook notebooks/
```
