# 🖼️ Image to LaTeX

## 📘 Introduction

This repository implements a deep learning model to solve the **Image-to-LaTeX** task: converting images of mathematical formulas into their corresponding LaTeX code. Inspired by the work of Guillaume Genthial (2017), this project explores various encoder-decoder architectures based on the Seq2Seq framework to improve the accuracy of LaTeX formula generation from images.

> 🧠 Motivation: Many students, researchers, and professionals encounter LaTeX-based documents but lack the ability to extract and reuse formulas quickly. This project aims to automate the conversion of math formula images into editable LaTeX.

---

## 🏗️ Model Architecture

Our model follows an **Encoder–Decoder with Attention** structure:

* **Encoder**: Several configurations based on CNNs (Convolutional Neural Networks), sometimes combined with a **Row Encoder (BiLSTM)** or **ResNet-18**.
* **Decoder**: A unidirectional **LSTM** network.
* **Attention**: Luong attention is used to enhance decoding accuracy.

Supported Encoder Variants:

* 🧱 Pure Convolution
* 🧱 Convolution + Row Encoder (BiLSTM)
* 🧱 Convolution + Batch Normalization
* 🧱 ResNet-18
* 🧱 ResNet-18 + Row Encoder (BiLSTM)

<p align="center">
  <img src="https://deforani.sirv.com/Images/Github/Image2Latex/image2latex.png" alt="Architecture Diagram"/>
</p>

---

## 📊 Dataset

### 🗂️ [im2latex-100k](https://www.kaggle.com/datasets/shahrukhkhan/im2latex100k)

* A commonly used dataset for benchmarking Image-to-LaTeX models.
* Preprocessed version available: [im2latex-sorted-by-size](https://www.kaggle.com/datasets/tuannguyenvananh/im2latex-sorted-by-size)

### 🗂️ [im2latex-170k](https://www.kaggle.com/datasets/rvente/im2latex170k)

* A larger dataset with more complex LaTeX expressions.
* Extended version with metadata: [im2latex-170k-meta-data](https://www.kaggle.com/datasets/tuannguyenvananh/im2latex-170k-meta-data)

---

## 🚀 How to Run

### Step 1: Install Requirements

Make sure you have the necessary packages installed.

```bash
pip install -r requirements.txt
```

### Step 2: Setup Weights & Biases (Optional for logging)

```bash
wandb login <your-wandb-key>
```

### Step 3: Training Example

```bash
python main.py \
    --batch-size 2 \
    --data-path C:\Users\nvatu\OneDrive\Desktop\dataset5\dataset5 \
    --img-path C:\Users\nvatu\OneDrive\Desktop\dataset5\dataset5\formula_images \
    --dataset 170k \
    --val \
    --decode-type beamsearch
```

---

## ✅ Results

From experiments using the **IM2LATEX-100k** dataset, the best-performing architecture was:

* **Convolutional Feature Encoder + BiLSTM Row Encoder**
* Achieved **77% BLEU-4 score**

---

## 📌 Notebooks

Explore pre-trained model performance and evaluation in Kaggle Notebooks:

* 🔍 [ResNet + Row Encoder Results](https://www.kaggle.com/code/tuannguyenvananh/image2latex-resnetbilstm-lstm)

---

## 📈 Future Work

* Integrating Transformer-based decoders
* Exploring pretrained vision encoders like ViT
* Improving performance on noisy or low-resolution images

---

## 📧 Author

**Nguyễn Văn Anh Tuấn**
📍 IUH - Industrial University of Ho Chi Minh City
✉️ [nvatuan3@gmail.com](mailto:nvatuan3@gmail.com)
