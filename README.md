# EMCFsys

**Towards foundation models for EM images analysis.** `emcfsys` provides a comprehensive toolkit and a [napari](https://github.com/napari/napari) plugin for Electron Microscopy (EM) image restoration and segmentation, featuring **EMCellFiner** and **EMCellFound**—two foundation models pre-trained on a massive dataset of 4 million EM images.

---

## ✨ Key Features

### 1. Key Features for emcfsys

* **Pre-trained Backbones**: Optimized on **4M+** EM images for superior feature extraction: EMCellFound and EMCellFiner.
* **Code-Free Train and Inference**: Train the segmentation model using Pre-trained Backbones and inference in the GUI without Any Code.
* **Code-Free and Training-Free Image restoration/super-resolution pipline**: Training-Free pipline for EM image restoration/super-resolution.

### 2. Segmentation pipline using Foundation model

* **Pre-trained Backbones**:
  * **EMCellFound Core**: Our foundational backbones are built upon state-of-the-art **ViT (Vision Transformer)** and **ConvNext** architectures. These models are pre-trained using advanced self-supervised learning frameworks, specifically **MAE (Masked Autoencoders)** and **DINOv3**, ensuring robust feature representation for complex electron microscopy data.
  * **Continuous Evolution**: We are committed to the iterative refinement of our models. We periodically retrain **EMCellFound** using superior architectures, optimized algorithms, and larger-scale datasets to ensure the system consistently delivers peak performance.
  * **Timm Library Integration**: To provide maximum flexibility, the system fully supports a wide range of popular pre-trained models from the **timm** library, allowing users to select the most suitable backbone for their specific research needs.
* **Segmentation Heads**: Includes **U-Net**, **PSPNet**, **DeepLabv3+**, and **UperNet**.
* **Finetune Models**: We support to finetune the EMCellFound/Timm-model to make specialize segmentation pipline.
* **Inference 2D/3D images**: We support to load the Checkpoint and inference image in 2D and 3D.
* **Tailored Training Strategies**: Detailed specifications of our training configurations can be found in [Functions notebook](notebooks\Functions.md). Key components include:
  * **Data Augmentation**: Robust `Dataset` class with multiple transform strategies.
  * **Loss Functions**: Integrated **CrossEntropy** and **Dice Loss** (Focal Loss coming soon).
  * **Metrics**: Real-time evaluation using **IoU**, **Accuracy**, and **F1-Score**.
  * **Smart Checkpoints**: Automatically preserves the best-performing model (Best IoU) and prunes redundant files.

### 3. Image restoration/super-resolution pipline using Foundation model

* **Retraining-free**: We train the image  restoration/super-resolution model EMCellFiner on 4M+ EM images, thus EMCellFiner has robust performance，can restore/super-resolution for most of EM images and make them finer.
* **Single-image**: We support restore/super-resolution in the GUI using GPU/CPU, and show in the GUI.
* **Multi-image**: We also support to restore/super-resolution the images in the folder, and output to another folder.

### 4. Tools

* **Annotation Support**: Built-in utility to convert **Labelme** JSON annotations to Semantic Segmentation masks.

---

## 📖Installation

To leverage the full potential of the EMCellFiner and EMCellFound foundation models, an **NVIDIA GPU**(We suggest > RTX3090) is highly recommended.

#### 1. Environment Preparation

We recommend using Conda/miniconda to create an isolated environment.
We suggest using python>=3.11 (We have successfully test the pipline in python version 3.8\3.9\3.10\3.11)

```
# Create a new environment named 'emcfsys' with Python> 3.11
conda create -n emcfsys python=3.11 -y
# Activate the environment
conda activate emcfsys
```

#### 2. Install PyTorch with GPU Support

emcfsys requires PyTorch > 1.3. For optimal performance, we recommend PyTorch 2.0+. Choose the command matching your CUDA version:

```
# For Linux and Window:
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
```

Or follow link to select the torch version [https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/)

#### 3. Install napari and emcfsys

3.1 You can follow the napari GUI installation document:[https://github.com/napari/napari](https://github.com/napari/napari)
Or use the follow pipline

```
pip install "napari[pyqt6, optional]" 
```

Then you can install`emcfsys` via [pip]:

```
pip install emcfsys
```

You can also install emcfsys in the napari-plugin-store

---

## 📖 Quick Start

1. Use as a napari Plugin
2. Launch napari: napari
3. Navigate to: Plugins -> emcfsys
4. Load your image and select EMCellFiner for instant enhancement.

## Other

## License

Distributed under the terms of the [GNU GPL v3.0] license,
"emcfsys" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[![License GNU GPL v3.0](https://img.shields.io/pypi/l/emcfsys.svg?color=green)]

Towards foundation models for EM images analysis. EMCellFiner and EMCellFound are two foundation models trained based on a 4 million EM images dataset.

---

[napari]: [https://github.com/napari/napari](https://github.com/napari/napari)
[napari-plugin-template]: [https://github.com/napari/napari-plugin-template](https://github.com/napari/napari-plugin-template)
