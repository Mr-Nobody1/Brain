# 🧠 Brain Tumor vs Breast Histopathology Classification

A comprehensive deep learning project for binary classification between MRI brain scans and breast histopathology images using state-of-the-art computer vision techniques.

## 📊 Results Summary

🎯 **Target Achievement**: ✅ **100% Test Accuracy** (Target: 95%)

| Metric | Score |
|--------|-------|
| **Test Accuracy** | **100.0%** |
| **Precision (Macro)** | **100.0%** |
| **Recall (Macro)** | **100.0%** |
| **F1-Score (Macro)** | **100.0%** |
| **ROC-AUC** | **100.0%** |
| **PR-AUC** | **100.0%** |

## 🖼️ Model Performance Visualizations

### Training History
![Training History](outputs/training_history_efficientnet_b0.png)

### Confusion Matrix
![Confusion Matrix](reports/confusion_matrix_detailed_efficientnet_b0.png)

### ROC & Precision-Recall Curves
![ROC and PR Curves](reports/roc_pr_curves_efficientnet_b0.png)

## 📋 Project Overview

This project implements a robust binary classification pipeline to distinguish between:
- **MRI Brain Scans**: 3,264 images from brain tumor datasets
- **Breast Histopathology**: 10,000 histopathological image patches

### 🏗️ Architecture & Approach

- **Model**: EfficientNet-B0 (pretrained on ImageNet)
- **Training Strategy**: Two-stage transfer learning
  - Stage 1: Train classifier head only (5 epochs)
  - Stage 2: Fine-tune entire model (8 additional epochs)
- **Image Size**: 224×224 pixels
- **Preprocessing**: ImageNet normalization, 3-channel conversion
- **Data Augmentation**: Comprehensive augmentation pipeline for training

## 📊 Dataset Information

### Dataset Splits
| Split | Total Images | MRI | Breast Histo | Split % |
|-------|-------------|-----|--------------|---------|
| **Train** | 9,284 | 2,284 (24.6%) | 7,000 (75.4%) | 70.0% |
| **Validation** | 1,989 | 489 (24.6%) | 1,500 (75.4%) | 15.0% |
| **Test** | 1,991 | 491 (24.7%) | 1,500 (75.3%) | 15.0% |
| **Total** | **13,264** | **3,264 (24.6%)** | **10,000 (75.4%)** | **100%** |

### Class Balance
- **Class Distribution**: ~25% MRI, ~75% Breast Histopathology
- **Balanced Evaluation**: All metrics calculated with both macro and weighted averaging

## 🚀 Key Features

### 1. **Comprehensive Preprocessing Pipeline**
- ✅ Multi-format image support (JPEG, PNG, TIFF, etc.)
- ✅ Automatic 3-channel conversion
- ✅ Robust error handling for corrupted images
- ✅ ImageNet normalization for transfer learning

### 2. **Advanced Data Augmentation**
- ✅ Geometric: Horizontal/vertical flips, rotations, random crops
- ✅ Photometric: Brightness/contrast adjustments
- ✅ Quality: Gaussian noise, motion blur, Gaussian blur
- ✅ Validation-specific: No augmentation for fair evaluation

### 3. **Training Optimizations**
- ✅ Two-stage transfer learning
- ✅ Cosine annealing learning rate schedule
- ✅ Early stopping with patience
- ✅ Best model checkpointing
- ✅ Comprehensive metrics tracking

### 4. **Evaluation & Reporting**
- ✅ Detailed confusion matrix analysis
- ✅ ROC and Precision-Recall curves
- ✅ Per-class and macro-averaged metrics
- ✅ Error analysis (no errors found!)
- ✅ Full reproducibility documentation

## 📁 Project Structure

```
Brain/
├── 📔 notebooks/
│   ├── 01_data_exploration_analysis.ipynb     # Dataset exploration
│   ├── 02_exploratory_data_analysis.ipynb     # EDA and visualizations
│   ├── 03_data_preparation_preprocessing.ipynb # Data preprocessing
│   └── 04_model_training_binary_classification.ipynb # Training pipeline
├── 📊 data/
│   ├── raw/                                   # Original datasets
│   └── processed/                             # Preprocessed data splits
├── 🤖 models/
│   └── best_efficientnet_b0_binary.pth       # Best trained model
├── 📈 outputs/                                # Training outputs
│   ├── training_history_efficientnet_b0.png
│   ├── confusion_matrix_efficientnet_b0.png
│   ├── roc_curve_efficientnet_b0.png
│   └── final_results_efficientnet_b0.json
├── 📋 reports/                                # Comprehensive evaluation
│   ├── confusion_matrix_detailed_efficientnet_b0.png
│   ├── roc_pr_curves_efficientnet_b0.png
│   ├── final_report_efficientnet_b0.json
│   ├── metrics_summary_efficientnet_b0.csv
│   └── error_analysis_efficientnet_b0.csv
├── 🐍 src/                                   # Source code modules
└── 📖 README.md                              # This file
```

## 🛠️ Technical Implementation

### Model Architecture
- **Backbone**: EfficientNet-B0 (5.3M parameters)
- **Classifier**: Custom binary classification head
- **Output**: Single neuron with BCEWithLogitsLoss
- **Activation**: Sigmoid for probability output

### Training Configuration
```python
CONFIG = {
    'model_name': 'efficientnet_b0',
    'image_size': 224,
    'batch_size': 32,
    'learning_rate': 3e-4,
    'weight_decay': 1e-4,
    'epochs': 50,
    'freeze_epochs': 5,
    'scheduler_type': 'cosine',
    'save_best_metric': 'val_f1'
}
```

### Hardware & Environment
- **Framework**: PyTorch 2.7.1+cu118
- **Device**: CUDA-enabled GPU
- **Training Time**: ~13 epochs (early stopping)
- **Best Epoch**: 13

## 📈 Performance Analysis

### Perfect Classification Achievement
The model achieved **perfect classification** on the test set:
- **Zero False Positives**: No MRI images misclassified as Breast Histopathology
- **Zero False Negatives**: No Breast Histopathology images misclassified as MRI
- **100% Sensitivity**: Perfect detection of positive cases
- **100% Specificity**: Perfect rejection of negative cases

### Confusion Matrix
```
                Predicted
              MRI  BreastHisto
    MRI       491      0
BreastHisto    0    1500
```

## 🔬 Methodology

### 1. **Data Preparation**
- Source datasets: Brain tumor MRI scans + Breast histopathology patches
- Preprocessing: Resize, normalize, convert to 3-channel RGB
- Split: 70% train, 15% validation, 15% test

### 2. **Model Development**
- Transfer learning from ImageNet-pretrained EfficientNet-B0
- Two-stage training: freeze backbone → fine-tune entire model
- Comprehensive data augmentation for generalization

### 3. **Evaluation Strategy**
- Hold-out test set for final evaluation
- Multiple metrics: accuracy, precision, recall, F1, AUC
- Error analysis and sample visualization

## 🚀 Usage

### Prerequisites
```bash
pip install torch torchvision
pip install albumentations opencv-python
pip install scikit-learn matplotlib seaborn
pip install pandas numpy tqdm
```

### Running the Pipeline
1. **Data Exploration**: Run `01_data_exploration_analysis.ipynb`
2. **EDA**: Execute `02_exploratory_data_analysis.ipynb`
3. **Preprocessing**: Process data with `03_data_preparation_preprocessing.ipynb`
4. **Training**: Train model using `04_model_training_binary_classification.ipynb`

### Model Inference
```python
import torch
from torchvision import transforms
from PIL import Image

# Load trained model
model = torch.load('models/best_efficientnet_b0_binary.pth')
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Predict
image = Image.open('path/to/image.jpg')
input_tensor = transform(image).unsqueeze(0)
with torch.no_grad():
    output = torch.sigmoid(model(input_tensor))
    prediction = "BreastHisto" if output > 0.5 else "MRI"
```

## 📊 Reproducibility

All experiments are fully reproducible with:
- **Fixed Random Seeds**: 42 across all libraries
- **Deterministic Operations**: CUDA deterministic mode enabled
- **Version Control**: Exact library versions documented
- **Configuration Tracking**: All hyperparameters saved
- **Dataset Documentation**: Exact split counts recorded

## 🎯 Future Work

- [ ] **Multi-class Extension**: Expand to classify specific brain tumor types
- [ ] **Model Compression**: Optimize for deployment (quantization, pruning)
- [ ] **Cross-validation**: Implement k-fold validation for robustness
- [ ] **Ensemble Methods**: Combine multiple architectures
- [ ] **Interpretability**: Add attention visualizations and GradCAM
- [ ] **Real-time Inference**: Deploy as web service or mobile app

## 🏆 Key Achievements

✅ **Perfect Test Accuracy**: 100.0% classification accuracy  
✅ **Robust Pipeline**: Comprehensive preprocessing and augmentation  
✅ **Transfer Learning**: Successful application of pretrained models  
✅ **Documentation**: Full reproducibility and detailed reporting  
✅ **Visualization**: Professional-grade results presentation  
✅ **Error Analysis**: Systematic evaluation framework  

## 📝 Citation

If you use this work, please cite:

```bibtex
@misc{brain_vs_breast_classification_2025,
  title={Binary Classification of MRI Brain Scans vs Breast Histopathology Images},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/Brain}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**⭐ Star this repository if you found it helpful!**

*Last Updated: August 31, 2025*
