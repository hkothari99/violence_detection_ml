# 🚨 Violence Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Real-time violence detection in video streams using deep learning**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Model Architecture](#-model-architecture) • [Dataset](#-dataset)

</div>

---

## 📋 Overview

This project implements an **AI-powered violence detection system** that analyzes video streams in real-time to identify violent activities. Built with **MobileNetV2** architecture and trained on the Real Life Violence Dataset, the system provides:

- ✅ Real-time video processing and violence detection
- ✅ Automated Telegram alerts with screenshots
- ✅ Frame-by-frame analysis with configurable frame skipping
- ✅ Visual output with labeled detection results
- ✅ Lightweight model optimized for deployment

## 🎯 Features

### Core Capabilities
- **Real-Time Detection**: Process video streams with configurable frame skip rates
- **High Accuracy**: MobileNetV2-based model trained on 700 videos (16,030 frames)
- **Instant Alerts**: Automated Telegram notifications with location and timestamp
- **Visual Output**: Annotated video output with violence indicators
- **Flexible Input**: Support for multiple video formats (MP4, AVI, etc.)

### Technical Highlights
- **Model**: MobileNetV2 transfer learning with custom classification head
- **Input Size**: 128x128 RGB images
- **Preprocessing**: Image augmentation (flip, zoom, brightness, rotation)
- **Inference**: Deque-based temporal smoothing for stable predictions
- **Alert Threshold**: Configurable violence detection trigger (default: 10 frames)

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) GPU with CUDA support for faster training

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/hkothari99/violence_detection_ml.git
cd violence_detection_ml
```

2. **Install dependencies**
```bash
pip install tensorflow opencv-python numpy telepot pytz imgaug
```

3. **Verify installation**
```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
```

## 📖 Usage

### Quick Start - Violence Detection

Run violence detection on sample videos:

```python
python -c "exec(open('violence_pred.ipynb').read())"
```

Or use the notebook directly in Jupyter:
```bash
jupyter notebook violence_pred.ipynb
```

### Custom Video Processing

```python
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from collections import deque

# Load the pre-trained model
model = load_model('./model/modelnew.h5')

# Process your video
video_path = 'path/to/your/video.mp4'
output_path = './output/result.avi'

# Run detection (see violence_pred.ipynb for full implementation)
print_results(video_path, output_filename='result.avi', frame_skip=2)
```

### Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `frame_skip` | Process every Nth frame | 2 |
| `output_filename` | Name of output video | `v_output.avi` |
| `limit` | Max frames to process | None (all) |
| `location` | Location for alerts | "Pune" |

### Telegram Alert Setup

1. Create a Telegram bot via [@BotFather](https://t.me/botfather)
2. Get your bot token and chat ID
3. Update the credentials in `violence_pred.ipynb`:
```python
bot = telepot.Bot('YOUR_BOT_TOKEN')
bot.sendMessage('YOUR_CHAT_ID', message)
```

## 🧠 Model Architecture

### MobileNetV2 Base
- **Architecture**: MobileNetV2 (ImageNet pre-trained)
- **Input Shape**: (128, 128, 3)
- **Total Parameters**: ~2.3M
- **Trainable Parameters**: Custom classification head

### Classification Head
```
MobileNetV2 (frozen) → Flatten → Dense(256, ReLU) → Dropout(0.5) → Dense(1, Sigmoid)
```

### Training Details
- **Dataset**: Real Life Violence Dataset
- **Training Samples**: 700 videos → 16,030 frames
- **Split**: 70% train / 30% test
- **Augmentation**: Flip, zoom, brightness, rotation
- **Optimizer**: Adam
- **Loss**: Binary Cross-Entropy

## 📊 Dataset

The model is trained on the **Real Life Violence Dataset**:
- **Total Videos**: 2,000 (1,000 violence + 1,000 non-violence)
- **Used for Training**: 700 videos (350 per class)
- **Frame Extraction**: Every 7th frame with augmentation
- **Classes**: Binary (Violence / Non-Violence)

## 📁 Project Structure

```
violence_detection_ml/
├── model/
│   └── modelnew.h5              # Pre-trained MobileNetV2 model
├── videos/
│   ├── v1.mp4                   # Sample test video 1
│   └── v2.mp4                   # Sample test video 2
├── output/
│   ├── output_v1.avi            # Processed output video 1
│   ├── output_v2.avi            # Processed output video 2
│   └── v_output.avi             # Default output video
├── violence_pred.ipynb          # Inference notebook
├── mobilenetv2_model.ipynb      # Training notebook
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## 🔧 Training Your Own Model

To train the model from scratch:

1. **Prepare Dataset**
   - Download the [Real Life Violence Dataset](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset)
   - Place in `./Downloads/violencedataset/Real Life Violence Dataset/`

2. **Run Training Notebook**
```bash
jupyter notebook mobilenetv2_model.ipynb
```

3. **Training Steps**:
   - Data preprocessing and augmentation
   - MobileNetV2 transfer learning
   - Model compilation and training
   - Model evaluation and saving

## 📈 Performance

- **Accuracy**: Optimized for real-world violence detection
- **Speed**: ~30 FPS on CPU (with frame_skip=2)
- **Model Size**: 9.5 MB (deployment-ready)
- **Inference Time**: ~33ms per frame (128x128 input)

## 🛠️ Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'telepot'`
```bash
pip install telepot
```

**Issue**: Model loading fails
```bash
# Ensure TensorFlow version compatibility
pip install tensorflow==2.8.0
```

**Issue**: OpenCV display error
```python
# Comment out cv2.imshow() for headless environments
# cv2.imshow('Violence Detection', output)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **MobileNetV2**: [Sandler et al., 2018](https://arxiv.org/abs/1801.04381)
- **Real Life Violence Dataset**: Available on Kaggle
- **TensorFlow & Keras**: Deep learning framework
- **OpenCV**: Computer vision library

## 📧 Contact

**Project Maintainer**: hkothari99

**Repository**: [https://github.com/hkothari99/violence_detection_ml](https://github.com/hkothari99/violence_detection_ml)

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

Made with ❤️ using TensorFlow and OpenCV

</div>
