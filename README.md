# EMCFsys

**Towards foundation models for EM images analysis.** `emcfsys` provides a comprehensive toolkit and a [napari] plugin for Electron Microscopy (EM) image restoration and segmentation, featuring **EMCellFiner** and **EMCellFound**—two foundation models pre-trained on a massive dataset of 4 million EM images.



---

## ✨ Key Features

### 1. Foundation Models & Architectures
* **Pre-trained Backbones**: Optimized on 4M+ EM images for superior feature extraction: EMCellFound and EMCellFiner.
* **Timm Integration**: Support for any backbone from the `pytorch-image-models` library.
* **Segmentation Heads**: Includes **U-Net**, **PSPNet**, **DeepLabv3+**, and **UperNet**.
* **Image restoration/super-resolution pipline**: 

### 2. Segmentation pipline using Foundation model
* **Pre-trained Backbones**: EMCellFound is the pretrained backbone based ViT/ConvNext. We trained it by MAE/Dinov3 and select the best model. We 定期使用更好的架构、算法或更多的数据重新训练模型，以保证模型性能最佳。同时我们支持Timm库中的常用的预训练模型，以满足用户需求。
* **Segmentation Heads**: Includes **U-Net**, **PSPNet**, **DeepLabv3+**, and **UperNet**.
* **Finetune Models**: We support to finetune the EMCellFound/Timm-model to make 特定的segmentation pipline.
* **Inference 2D/3D images**: We support to load the Checkpoint and inference image in 2D and 3D.
* **定制化的训练策略**: 训练策略可以在Functions.md中找到.Including:
    * **Data Augmentation**: Robust `Dataset` class with multiple transform strategies.
    * **Loss Functions**: Integrated **CrossEntropy** and **Dice Loss** (Focal Loss coming soon).
    * **Metrics**: Real-time evaluation using **IoU**, **Accuracy**, and **F1-Score**.
    * **Smart Checkpoints**: Automatically preserves the best-performing model (Best IoU) and prunes redundant files.


### 3. Image restoration/super-resolution pipline using Foundation model
* **Reraining-free**: We train the image  restoration/super-resolution model EMCellFiner on 4M+ EM images, thus EMCellFiner has robust 性能，can restore\super-resolution for most of EM images and make them finer. 
* **Single-image**: We support restore/super-resolution in the GUI using GPU/CPU, and show in the GUI.
* **Multi-image**: We also support to restore/super-resolution the images in the folder, and output to another folder.


### 4. Tools
* **Annotation Support**: Built-in utility to convert **Labelme** JSON annotations to Semantic Segmentation masks.



---

## 📖Installation

To leverage the full potential of the EMCellFiner and EMCellFound foundation models, an **NVIDIA GPU**(We suggest it > RTX3090) is highly recommended.

---

#### 1. Environment Preparation
We recommend using Conda/miniconda to create an isolated environment. 
We suggest using python>3.11 (We have test the pipline in 3.8\3.9\3.10\3.11)

    ```
    # Create a new environment named 'emcfsys' with Python> 3.11
    conda create -n emcfsys python=3.11 -y

    # Activate the environment
    conda activate emcfsys
    ```

#### 2. Install PyTorch with GPU Support
emcfsys requires PyTorch > 1.3. For optimal performance, we recommend PyTorch 2.0+. Choose the command matching your CUDA version:

    ```
    pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
    ```
Or follow link to pip the torch [https://pytorch.org/get-started/previous-versions/]

#### 3. Install napari and emcfsys
    You can follow the napari GUI installation document[napari]: https://github.com/napari/napari. Or use the follow pipline

    ```
    pip install "napari[pyqt6, optional]"
    ```
    
    · Then you can install `emcfsys` via [pip]:

    ```
    pip install emcfsys
    ```
    · You can also install emcfsys in the napari-plugin-store



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

[napari]: https://github.com/napari/napari
[napari-plugin-template]: https://github.com/napari/napari-plugin-template

[![License GNU GPL v3.0](https://img.shields.io/pypi/l/emcfsys.svg?color=green)](https://github.com/Zhejiang University , github:yzy0102/emcfsys/raw/main/LICENSE)

Towards foundation models for EM images analysis. EMCellFiner and EMCellFound are two foundation models trained based on a 4 million EM images dataset.

---