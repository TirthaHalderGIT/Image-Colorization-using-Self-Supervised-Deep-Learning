# Image Colorization using Self-Supervised Deep Learning

This repository contains a complete deep learning pipeline for automatic colorization of grayscale images using a self-supervised denoising autoencoder built on a custom U-Net encoder–decoder architecture.  
The model predicts the chrominance (a, b) channels from only the luminance (L) channel in LAB color space, requiring no labeled color supervision.

## Key Features

- Self-supervised learning using Gaussian noise injected into the L-channel.
- Custom U-Net–based encoder–decoder model with skip connections for high-resolution colorization.
- Trained on the ImageNet-1K dataset (100k images) with LAB preprocessing.
- Mixed-precision training using PyTorch AMP and GradScaler.
- Resumable training via automatic checkpoint restoration (model, optimizer, history, epoch).
- Standalone inference script for colorizing any grayscale image locally.

## Model Architecture Overview

### Encoder
- Four ConvBlock + MaxPool stages.
- Channel progression: 64 → 128 → 256 → 512.

### Bottleneck
- 1024-channel ConvBlock.

### Decoder
- Four UpBlocks (ConvTranspose2D + skip connections + ConvBlock).
- Channel progression: 1024 → 512 → 256 → 128 → 64.

### Output Layer
- 1×1 convolution producing 2-channel output (a, b).

**ConvBlock structure:**
- Two 3×3 convolution layers
- Batch Normalization
- ReLU activation

## Folder Structure

```
project/
│
├── colorize_image.py               # inference script
├── colorization_sdae_unet_best.pth # trained model weights
│
├── notebook/                       # Kaggle training notebook
│
├── inputs/
│   └── sample_bw.jpg               # grayscale input
│
└── outputs/
    ├── sample_bw_colorized.png
    └── sample_bw_side_by_side.png
```

## Installation

```
pip install torch torchvision pillow matplotlib scikit-image numpy
```

## Running Inference

```
python colorize_image.py
```

## Training Details

- Dataset: ImageNet-1K (100,000 images)
- Color Space: LAB
- Input normalization:
  - L-channel → scaled to [-1, 1]
  - a/b channels → scaled to [-1, 1]
- Loss Function: L1 loss on (a, b)
- Optimizer: Adam
- Batch Size: 64
- Augmentations: RandomResizedCrop, CenterCrop, noise injection
- Checkpointing: automatic resume support

## Future Scope

- GAN-based adversarial colorization
- Real-time video colorization
- Interactive user-guided colorization
- Web or mobile deployment (ONNX/TFLite)
- Historical photo restoration tools
