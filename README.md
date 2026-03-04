---

# D-SVG: Diffusion-Based Slice-to-Volume Generation for Fetal Brain MRI

Official PyTorch implementation of

**D-SVG: Diffusion Based Slice-to-Volume Generation with Implicit Neural Representation for Fetal Brain MRI**

📄 Paper:

---

## 🧠 Overview

Reconstructing high-quality **3D fetal brain MRI volumes** from motion-corrupted **2D slice stacks** is essential for accurate prenatal diagnosis. However, fetal MRI acquisition is often affected by motion artifacts, inconsistent slice contrast, and incomplete brain extraction regions.

To address these challenges, we propose **D-SVG**, a novel framework that integrates:

* **Diffusion models** for anatomical prior generation
* **Implicit Neural Representation (INR)** for slice fidelity refinement

Unlike traditional slice-to-volume reconstruction methods, D-SVG formulates the task as **3D volume generation guided by scanned slices**, enabling more robust reconstruction under severe motion corruption. 

---

## 🚀 Method Overview

The D-SVG framework consists of two main components:

### 1️⃣ Diffusion-based Volume Prior Generation

A conditional **3D diffusion model** is used to generate a plausible brain volume prior.

Inputs:

* Noisy volume
* Estimated brain mask

The diffusion model learns anatomical priors from fetal brain volumes and generates structurally consistent templates.

---

### 2️⃣ Slice-Constrained Fidelity Refinement

An **Implicit Neural Representation (INR)** is introduced to enforce consistency between the generated volume and the scanned slices.

The refinement process contains three stages:

1. **Volume Prior Embedding**
2. **Slice Constraint Optimization**
3. **Volume Sampling and Noise Injection**

The INR network models voxel intensities as a continuous function, allowing reconstruction with arbitrary resolution. 

---

## 🏗️ Framework Pipeline

```
Motion-corrupted slices
        │
        ▼
Diffusion Model
(Volume Prior Generation)
        │
        ▼
Slice-Constrained Fidelity
(INR Optimization)
        │
        ▼
High-quality 3D Fetal Brain MRI
```

The diffusion process provides **global anatomical prior**, while INR ensures **slice-level consistency and detail preservation**.

---

## 📊 Experimental Results

Experiments were conducted on both **clinical fetal brain MRI** and **simulated datasets**.

The proposed method achieved superior performance compared with existing reconstruction methods:

| Method      | PSNR      | SSIM      | NCC       |
| ----------- | --------- | --------- | --------- |
| SVRTK       | 12.81     | 0.926     | 0.711     |
| NIFTYMIC    | 17.06     | 0.930     | 0.357     |
| SVR + SVoRT | 19.19     | 0.934     | 0.784     |
| NeSVoR      | 20.75     | 0.931     | 0.761     |
| **D-SVG**   | **21.69** | **0.940** | **0.866** |

D-SVG demonstrates improved structural consistency and reduced reconstruction artifacts. 

---

## 🗂 Repository Structure

```
dsvg
│
├── DMpipe.py
├── dataset.py
├── NeSVoR/
│
├── dataset/
│   ├── brats2021/
│   └── ...
│
├── LICENSE
└── README.md
```

---

## ⚙️ Installation

Create the environment and install dependencies:

```bash
pip install -r requirements.txt
```

Required libraries typically include:

* PyTorch
* NumPy
* Nibabel
* TorchIO

---

## 🧪 Dataset Preparation

D-SVG expects **multi-stack fetal MRI slices** as input.

Typical preprocessing steps include:

* brain extraction
* slice alignment
* mask estimation
* stack normalization

Example dataset structure:

```
dataset
 ├── subject1
 │    ├── stack1.nii.gz
 │    ├── stack2.nii.gz
 │    └── stack3.nii.gz
```

---

## 🎓 Training

Run training with:

```
python DMpipe.py
```

Training pipeline includes:

1. diffusion prior training
2. slice-constrained INR refinement
3. iterative reconstruction

---

## 🎨 Reconstruction

After training, reconstructed volumes can be generated using the trained model.

```
python DMpipe.py --mode inference
```

Output:

```
reconstructed_volume.nii.gz
```

---

## 📋 TODO

* [x] Release core implementation
* [ ] Release pretrained weights
* [ ] Add inference scripts
* [ ] Provide detailed training tutorial
* [ ] Release Docker environment

---

## 📜 Citation

If you find this work useful for your research, please cite:

```
@article{lv2025dsvg,
  title={D-SVG: Diffusion Based Slice-to-Volume Generation with Implicit Neural Representation for Fetal Brain MRI},
  author={Lv, Yao and Cai, Zhibao and Chen, Shengxian and Yang, Chaoxiang and Zhang, Xin},
  journal={},
  year={2025}
}
```

---

## 🙏 Acknowledgements

This project builds upon several excellent open-source works:

* NeSVoR
* SVRTK
* NIFTYMIC
* Denoising Diffusion Probabilistic Models

---

💡 **我建议你 README 里再加两块内容，会让 GitHub 项目更像顶会代码：**

1️⃣ **Method figure（论文 pipeline 图）**
2️⃣ **Reconstruction visualization**

类似这样：

```
images/
   pipeline.png
   reconstruction.png
```

---

如果你愿意，我可以帮你 **再升级一个 GitHub 论文级 README**，比如加入：

* pipeline 图展示
* reconstruction GIF
* citation badge
* arxiv badge
* project page

那一版基本就是 **MICCAI / CVPR 开源代码风格**。
