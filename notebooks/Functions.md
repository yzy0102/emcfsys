# EMCFsys Development Documentation

This document outlines the current functional status and future roadmap of the **EMCFsys** electron microscopy image segmentation system.

---

## 1. Core Segmentation Functions

The core engine supports pixel-wise classification tailored for high-resolution EM imagery.

### 🟢 Loss Functions

- [X] **CrossEntropy Loss**: Standard categorical voxel/pixel loss.
- [X] **Dice Loss**: Optimized for handling class imbalance in fine structures.
- [ ] **Focal Loss**: *Planned* - To address extreme foreground-background ratio issues.

### 🟢 Evaluation Metrics

- [X] **IoU (Intersection over Union)**: Primary accuracy metric.
- [X] **Accuracy**: Global pixel-level classification precision.
- [X] **F1-Score**: Balancing precision and recall for morphological analysis.

### 🟢 Checkpoint Management

- [X] **Dynamic Saving**: Automatically saves the best model based on IoU and prunes inferior checkpoints.
- [X] **Persistence**: Exports the final model state post-training.
- [ ] **Early Stopping**: *Planned* - Monitor validation loss to prevent overfitting.

### 🟢 Visualization

- [X] **Loss Curve**: Real-time plotting of training/validation loss.
- [ ] **IoU Trend**: *Planned* - Visualizing metric convergence over epochs.

---

## 2. Model Inference & Deployment

Modular support for model loading and real-time inference.

- [X] **Model Import**: Supports loading custom weights.
- [X] **Self-Contained Serialization**: Saved weights include the model architecture for "plug-and-play" inference.
- [X] **Inference Pipeline**: Automated processing from raw images to segmentation masks.

---

## 3. Dataset & Data Pipeline

Robust data handling for diverse EM imaging modalities.

- [X] **Advanced Transforms**: Integrated multiple data augmentation strategies (Rotation, Scaling, Noise, etc.).
- [ ] **Multi-Dataset Support**: *Planned* - Standardized input for heterogeneous data sources.
- [ ] **Annotation Flexibility**: *Planned* - Compatibility with multiple mask and vector formats.

---

## 4. Model Architectures

Leveraging state-of-the-art backbones and segmentation heads.

### 🏗️ Backbones

- [X] **Timm Library Integration**: Full support for `pytorch-image-models` backbones (ResNet, EfficientNet, ViT, etc.).

### 🏗️ Semantic Segmentation Heads

- [X] **U-Net**: Specifically optimized for biomedical/EM image contexts.
- [X] **PSPNet**: Pyramid Pooling Module for global scene parsing.
- [X] **DeepLabv3+**: Utilizing Atrous Convolution for multi-scale feature capture.
- [X] **UperNet**: Unified Perceptual Parsing for comprehensive feature extraction.

---

## 5. Pre-processing Tools

Bridging the gap between manual labeling and training.

- [X] **Labelme Integration**: Integrated utility for converting `Labelme` JSON annotations into standard Semantic Segmentation ground-truth masks.

---

## 🛠 Setup & Contribution

*For information on how to set up the development environment or contribute to the roadmap, please refer to [CONTRIBUTING.md].*
