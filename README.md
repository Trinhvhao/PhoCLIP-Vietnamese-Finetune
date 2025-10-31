<div align="center">

# PhoCLIP - Vietnamese Image-Text Matching

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**A Vietnamese CLIP model for image-text retrieval using PhoBERT and ResNet50**

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Model Architecture](#model-architecture) • [Training](#training) • [Results](#results)

</div>

---

## 📋 Overview

PhoCLIP is a Vietnamese adaptation of the CLIP (Contrastive Language-Image Pre-training) model, designed specifically for Vietnamese image-text matching tasks. It combines:

- **Text Encoder**: PhoBERT-large (Vietnamese BERT)
- **Image Encoder**: ResNet50 (pretrained on ImageNet)
- **Projection Heads**: Maps both modalities to a shared embedding space

## ✨ Features

- 🇻🇳 **Vietnamese Language Support** - Optimized for Vietnamese text with PhoBERT
- 🖼️ **Multi-Dataset Training** - Trained on COCO, Flickr, KTVIC, and OpenViIC datasets
- 🔍 **Bidirectional Search** - Find images from text or text from images
- ⚡ **Fast Inference** - Optimized for GPU acceleration
- 📊 **High Accuracy** - Achieves competitive Top-5 accuracy on Vietnamese datasets

## 🚀 Installation

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# CUDA-enabled GPU (recommended)
nvidia-smi
```

### Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install timm transformers
pip install py_vncorenlp
pip install pandas numpy opencv-python albumentations
pip install matplotlib seaborn tqdm
pip install jsonlines
```

### Download VnCoreNLP

```python
import py_vncorenlp
py_vncorenlp.download_model()
```

## 📖 Usage

### Quick Start

```python
import torch
from phoclip import CLIPModel, find_matches
from transformers import AutoTokenizer

# Load model
model = CLIPModel()
model.load_state_dict(torch.load("best.pt"))
model.eval()

# Search images by text
find_matches(
    model,
    image_embeddings,
    image_filenames,
    text="xe hơi đậu trước ngôi nhà",
    n=25
)
```

### Training from Scratch

```python
from phoclip import main

# Configure training parameters in CFG class
# Then run training
main()
```

### Extract Embeddings

```python
from phoclip import get_image_embeddings, get_text_embeddings

# Get image embeddings
model, img_embeds = get_image_embeddings(valid_df, "best.pt")

# Get text embeddings
model, txt_embeds = get_text_embeddings(valid_df, "best.pt")
```

## 🏗️ Model Architecture

```
┌─────────────────────────────────────────────────────┐
│                   PhoCLIP Model                     │
├─────────────────────────────────────────────────────┤
│                                                     │
│   ┌──────────────┐         ┌──────────────┐         │
│   │ Image Input  │         │  Text Input  │         │
│   │  (224x224)   │         │  (max_len)   │         │
│   └──────┬───────┘         └──────┬───────┘         │
│          │                        │                 │
│          ▼                        ▼                 │
│   ┌──────────────┐         ┌──────────────┐         │
│   │  ResNet50    │         │PhoBERT-large │         │
│   │  (2048-dim)  │         │  (1024-dim)  │         │
│   └──────┬───────┘         └──────┬───────┘         │
│          │                        │                 │
│          ▼                        ▼                 │
│   ┌──────────────┐         ┌──────────────┐         │
│   │ Projection   │         │ Projection   │         │
│   │    Head      │         │    Head      │         │
│   │  (512-dim)   │         │  (512-dim)   │         │
│   └──────┬───────┘         └──────┬───────┘         │
│          │                        │                 │
│          └──────────┬─────────────┘                 │
│                     ▼                               │
│          ┌──────────────────────┐                   │
│          │  Contrastive Loss    │                   │
│          │  (Temperature=1.0)   │                   │
│          └──────────────────────┘                   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## 🎯 Training

### Configuration

Key hyperparameters in `CFG` class:

```python
batch_size = 64
epochs = 1
image_encoder_lr = 1e-4
text_encoder_lr = 1e-5
head_lr = 1e-3
temperature = 1.0
projection_dim = 512
max_length = 70
```

### Datasets

The model is trained on multiple Vietnamese image-caption datasets:

| Dataset | Images | Captions | Language |
|---------|--------|----------|----------|
| COCO (translated) | 123,287 | 616,767 | Vietnamese |
| Flickr30k | 31,783 | 158,915 | Vietnamese |
| KTVIC | 4,327 | 21,635 | Vietnamese |
| OpenViIC | 9,328 | 46,640 | Vietnamese |

### Training Pipeline

1. **Data Preprocessing**
   - Image resizing to 224x224
   - Vietnamese text segmentation with VnCoreNLP
   - Normalization

2. **Model Training**
   - Contrastive learning with symmetric cross-entropy loss
   - AdamW optimizer with learning rate scheduling
   - Early stopping based on validation loss

3. **Evaluation**
   - Top-K accuracy (K=1, 5, 10)
   - Image-to-text and text-to-image retrieval

## 📊 Results

### Top-K Accuracy

| Metric | Score |
|--------|-------|
| Top-1 | TBD |
| Top-5 | TBD |
| Top-10 | TBD |

### Example Queries

**Text Query**: "xe hơi đậu trước ngôi nhà"
- Successfully retrieves images of cars parked in front of houses

**Image Query**: Upload an image
- Returns similar images from the database

## 🔧 Advanced Usage

### Custom Dataset

```python
import pandas as pd

# Prepare your data
df = pd.DataFrame({
    'image': ['img1.jpg', 'img2.jpg'],
    'caption': ['mô tả 1', 'mô tả 2']
})

# Build dataloader
train_loader = build_loaders(df, tokenizer, mode="train")
```

### Fine-tuning

```python
# Freeze image encoder
CFG.image_encoder_trainable = False

# Train only text encoder and projection heads
model = CLIPModel()
# ... training code
```

## 📁 Project Structure

```
Phoclip/
├── phoclip.py          # Main implementation
├── phoclip.ipynb       # Jupyter notebook
├── README.md           # This file
├── best.pt             # Trained model weights (after training)
└── images/             # Image dataset directory
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **PhoBERT**: [VinAI Research](https://github.com/VinAIResearch/PhoBERT)
- **CLIP**: [OpenAI](https://github.com/openai/CLIP)
- **VnCoreNLP**: [Vietnamese NLP Toolkit](https://github.com/vncorenlp/VnCoreNLP)
- **Original PhoCLIP**: [ducngg/PhoCLIP](https://github.com/ducngg/PhoCLIP)
- **Datasets**: COCO, Flickr30k, KTVIC, OpenViIC

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

<div align="center">

Made with ❤️ for Vietnamese NLP

</div>
