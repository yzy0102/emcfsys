# EMCFsys

**Towards foundation models for EM images analysis.** `emcfsys` provides a comprehensive toolkit and a [napari](https://github.com/napari/napari) plugin for Electron Microscopy (EM) image restoration and segmentation, featuring **EMCellFiner** and **EMCellFound**—two foundation models pre-trained on a massive dataset of 4 million EM images.

---

# ✨ Key Features

### 1. Key Features for emcfsys

* **Pre-trained Backbones**: Optimized on **4M+** EM images for superior feature extraction: EMCellFound and EMCellFiner.
* **Code-Free Train and Inference**: Train the segmentation model using Pre-trained Backbones and inference in the GUI without Any Code.
* **Code-Free and Training-Free Image restoration/super-resolution pipline**: Training-Free pipline for EM image restoration/super-resolution.
* Code-Free GUI for Napari, and Code-based example tutorial in [demo_notebook.ipynb](demo_notebook.ipynb). This notebook is built upon mmengine, mmseg, and mmdet to facilitate code‑driven training workflows.
* **Fixed bugs caused by missing configs for mmlab‑backend.**

### 2. Segmentation pipline using Foundation model

* **Pre-trained Backbones**:
  * **EMCellFound Core**: Our foundational backbones are built upon state-of-the-art **ViT (Vision Transformer)** and **ConvNext** architectures. These models are pre-trained using advanced self-supervised learning frameworks, specifically **MAE (Masked Autoencoders)** and **DINOv3**, ensuring robust feature representation for complex electron microscopy data.
  * **Continuous Evolution**: We are committed to the iterative refinement of our models. We periodically retrain **EMCellFound** using superior architectures, optimized algorithms, and larger-scale datasets to ensure the system consistently delivers peak performance.
  * **Timm Library Integration**: To provide maximum flexibility, the system fully supports a wide range of popular pre-trained models from the **timm** library, allowing users to select the most suitable backbone for their specific research needs.
* **Segmentation Heads**: Includes **U-Net**, **PSPNet**, **DeepLabv3+**, **UperNet** , **Mask2former**.
* **Finetune Models**: We support to finetune the EMCellFound/Timm-model to make specialize segmentation pipline.
* **Inference 2D/3D images**: We support to load the Checkpoint and inference image in 2D and 3D.
* **Tailored Training Strategies**: Detailed specifications of our training configurations can be found in the [English user guide](docs/EMCFsys_UserGuide_English.md). Key components include:
  * **Data Augmentation**: Robust `Dataset` class with multiple transform strategies.
  * **Loss Functions**: Integrated **CrossEntropy** and **Dice Loss** (Focal Loss coming soon).
  * **Metrics**: Real-time evaluation using **IoU**, **Accuracy**, and **F1-Score**.
  * **Smart Checkpoints**: Automatically preserves the best-performing model (Best IoU) and prunes redundant files.

### 3. Image restoration/super-resolution pipline using Foundation model

* **Retraining-free**: We train the image restoration/super-resolution model EMCellFiner on 4M+ EM images, thus EMCellFiner has robust performance，can restore/super-resolution for most of EM images and make them finer.
* **Single-image**: We support restore/super-resolution in the GUI using GPU/CPU, and show in the GUI.
* **Multi-image**: We also support to restore/super-resolution the images in the folder, and output to another folder.

### 4. Tools

* **Model Manager & JSON‑based Configuration**: Quickly load all parameters within the Napari GUI for model fine‑tuning.
* **Dataset‑Converter Support**: Built‑in utility to convert Labelme JSON annotations into semantic‑segmentation masks and instance‑segmentation JSON files.

---

## 📖 Installation

An NVIDIA GPU is strongly recommended for EMCellFiner inference and EMCellFound fine-tuning. The validated Windows environment uses Python 3.11.10 and PyTorch 2.5.1 with CUDA 11.8. The portable Linux package snapshot is derived from a working environment that used PyTorch 2.4.1 with CUDA 12.1.

### 1. Choose the installation type

- **napari GUI only:** install PyTorch, napari, and the released `emcfsys` package. This is sufficient for the graphical restoration, training, and inference workflows.
- **MMLab demo and paper workflows:** clone this repository and use the pinned environment files. This installation includes MMCV, MMEngine, MMSegmentation, and MMDetection for the examples in [demo_notebook.ipynb](demo_notebook.ipynb).

Windows and Linux use separate pinned environment snapshots because their tested PyTorch, CUDA, and MMCV builds differ. The other main operating-system differences are the compiler, environment-variable syntax, and Linux GUI libraries.

| Requirement               | Windows                                                       | Linux                                                                    |
| ------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------ |
| C/C++ compiler            | Visual Studio Build Tools with Desktop development with C++   | GCC/G++ through`build-essential`                                       |
| CUDA Toolkit for MMCV ops | CUDA Toolkit 11.8 and`CUDA_HOME` when MMCV must be compiled | CUDA Toolkit 12.1,`nvcc`, and `CUDA_HOME` when MMCV must be compiled |
| napari GUI                | Native Windows desktop                                        | X11 or Wayland graphical session and OpenGL libraries                    |
| Pinned requirements       | `pined_requirement.txt`                                     | `pined_requirement_linux.txt`                                          |
| Path style                | `path\to\file`                                              | `path/to/file`                                                         |

The Windows file records the validated Windows package stack. The Linux file replaces local editable source checkouts with compatible PyPI releases to make installation portable. MMCV binary compatibility still depends on the CUDA Toolkit, compiler, NVIDIA driver, and GPU architecture of the host system.

### 2. Create the Conda environment

Use the same Python version on both operating systems:

```bash
conda create -n EMCF_napari python=3.11.10 -y
conda activate EMCF_napari
```

### 3. Install the napari GUI only

The following commands apply to both Windows and Linux:

Just choose pytorch and cuda version suited for your computer:

pytorch from 1.13.0 to 2.7.0 and cuda from 11.3 to12.8 is ok.

We recommend pytorch==2.4.1/2.5.1 , cuda ==11.8/12.1

```Shell

python -m pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 torchaudio==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# In China, speed uo to install
python -m pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 torchaudio==2.5.1+cu118 --index-url https://mirror.nju.edu.cn/pytorch/whl/cu118


python -m pip install napari
python -m pip install emcfsys

python -m pip install frc -i https://pypi.org/simple
```

The napari plugin can then be started with:

```bash
napari
```

Linux users need a graphical X11 or Wayland session to open napari. On a headless Linux server, use the code-based workflows without launching the GUI, or configure an appropriate remote display.

### 4. Install the pinned MMLab environment on Windows and Linux

Run the following commands in PowerShell from the cloned repository root:

```powershell

python -m pip install ninja==1.13.0 psutil==7.1.3 wheel==0.45.1


python -m pip install --no-build-isolation -r pined_requirement.txt
```

### 5. Verify the MMLab installation

Run this check on either operating system:

```bash
python -c "import torch, mmcv, mmengine, mmseg, mmdet; from mmcv.ops import MultiScaleDeformableAttention; print(torch.__version__, mmcv.__version__, mmengine.__version__, mmseg.__version__, mmdet.__version__)"
```

The expected Windows core versions are:

```text
PyTorch 2.5.1+cu118
MMCV 2.0.0rc4
MMEngine 0.10.7
MMSegmentation 1.2.2
MMDetection 3.3.0
```

The expected Linux core versions are:

```text
PyTorch 2.4.1+cu121
MMCV 2.1.0
MMEngine 0.10.7
MMSegmentation 1.2.2
MMDetection 3.3.0
```

After this check succeeds, open [demo_notebook.ipynb](demo_notebook.ipynb) to run the classification, semantic-segmentation, instance-segmentation, and image-restoration examples. Use repository-relative paths or `pathlib.Path` when adapting notebook paths between Windows and Linux.

---

## 📖 Quick Start

1. Use as a napari Plugin
2. Launch napari: napari
3. Navigate to: Plugins -> emcfsys
4. Load your image and select EMCellFiner for instant enhancement.

## 📖 Tutorial

All tutorials and feature descriptions can be found in the [tutorial documentation](./docs/EMCFsys_UserGuide_English.md) (click me!).

## License

Distributed under the terms of the [GNU GPL v3.0] license,
"emcfsys" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[![License GNU GPL v3.0](https://img.shields.io/pypi/l/emcfsys.svg?color=green)]

Towards foundation models for EM images analysis. EMCellFiner and EMCellFound are two foundation models trained based on a 4 million EM images dataset.

If you use this project, please cite:

Yu Z, Guo J, Liu F, et al. EMCF ecosystem: Towards pretrained foundation model for electron microscopy image analysis[J]. bioRxiv, 2025: 2025.12. 09.693109.

---

[napari]: [https://github.com/napari/napari](https://github.com/napari/napari)
[napari-plugin-template]: [https://github.com/napari/napari-plugin-template](https://github.com/napari/napari-plugin-template)
