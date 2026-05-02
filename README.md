# CS731 — Emotion-Aware Study Assistant Chatbot

A real-time emotion-aware chatbot that detects facial expressions via webcam and adapts its responses to support students during study sessions.

---

## Team Members
- Harini Selvaraj
- Gem Ann Joseph
- Jamuna Rani Pandia Rajan

---

## Project Overview

This system combines:
- **Real-time facial emotion detection** using YOLOv8 face detection + EfficientNet-B0 classifier
- **Emotion-aware chatbot** powered by Claude Haiku 4.5 via OpenRouter API
- **Desktop GUI** built with PyQt5 supporting both voice and text input

When a student's emotion changes (e.g. from neutral to sad), the chatbot automatically adapts its tone and responses to provide appropriate support.

---

## Project Structure

```
cs731_project/
├── train.py                  # Model training script (3 CNN models)
├── dataset.py                # Custom dataset loader
├── inference.py              # Emotion inference helper
├── gui.py                    # Main PyQt5 desktop application
├── chatbot.py                # Emotion-aware chatbot (OpenRouter API)
├── yolov8_face.py            # Standalone webcam demo with YOLOv8
├── yolov8n-face.pt           # YOLOv8 face detection weights
├── .env                      # API key (not committed to git)
├── checkpoints/
│   └── efficientnet/
│       └── best.pt           # Best trained model checkpoint
├── train_images/             # Training dataset (28,953 images)
│   ├── anger/
│   ├── contempt/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test_images/              # Test dataset (232 images)
    └── ...
```

---

## Emotion Classes

The system recognises 8 emotions based on Ekman's theory:

| Emotion | Student Context |
|---------|----------------|
| Happy | Understanding topic well |
| Sad | Personal issues affecting study |
| Angry | Frustrated with subject material |
| Fear | Exam anxiety or deadline stress |
| Surprise | Something clicked or unexpected result |
| Disgust | Disliking the topic or approach |
| Contempt | Bored or disengaged |
| Neutral | Normal focused study state |

---

## Model Comparison Results

| Model | Test Accuracy | Notes |
|-------|-------------|-------|
| ConvNeXtV2 Pico | ~78.8% | Lecturer baseline |
| ResNet-50 | 81.90% | Good improvement |
| **EfficientNet-B0** | **82.76%** | **Best — selected** |

Dataset: AffectNet-HQ + RAF-DB combined (28,953 train / 232 test images)

---

## Setup Instructions

### 1. Prerequisites
- Ubuntu 22.04 (WSL2 or native)
- Anaconda
- NVIDIA GPU (recommended)

### 2. Activate Environment
```bash
conda activate cs731
```

### 3. Install Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install timm ultralytics PyQt5 opencv-python-headless
pip install SpeechRecognition pyaudio python-dotenv pyttsx3
```

### 4. Set Up API Key
Create a `.env` file in the project root:
```
OPENROUTER_API_KEY=your_key_here
```

Get your free API key at: https://openrouter.ai

### 5. Run the Application
```bash
cd cs731_project
python gui.py
```

---

## How to Run

### Main GUI (recommended)
```bash
python gui.py
```

### Terminal Chatbot Test
```bash
python chatbot.py
```

### Webcam Emotion Demo Only
```bash
python yolov8_face.py
```

### Train a Model
```bash
# Edit MODEL_NAME in train.py first:
# 'convnextv2', 'resnet50', or 'efficientnet'
python train.py
```

---

## GUI Features

- **Left panel** — Live webcam feed with face bounding box and emotion label
- **Right panel** — Chat interface with conversation history
- **Text input** — Type messages and press Enter or click Send
- **Voice input** — Click 🎤 microphone button to speak
- **Auto emotion response** — Chatbot automatically responds when emotion changes
- **Clear chat** — Reset conversation history

---

## Configuration

Key settings at top of `gui.py`:

```python
MODEL_PATH       = "checkpoints/efficientnet/best.pt"  # Emotion model
YOLO_PATH        = "yolov8n-face.pt"                   # Face detector
WEBCAM_INDEX     = 0                                    # Camera index
EMOTION_INTERVAL = 5                                    # Detection frequency
SMOOTH_WINDOW    = 8                                    # Smoothing window
```

Key settings at top of `train.py`:

```python
MODEL_NAME    = 'efficientnet'   # Model to train
BATCH_SIZE    = 32               # Training batch size
LEARNING_RATE = 3e-4             # Learning rate
NUM_EPOCHS    = 30               # Maximum epochs
PATIENCE      = 5                # Early stopping patience
```

---

## LLM Used

| Model | Provider | Notes |
|-------|----------|-------|
| Claude Haiku 4.5 | Anthropic via OpenRouter | Selected for speed + quality |

---

## References

- Ekman, P. (1992). An argument for basic emotions
- AffectNet Dataset — Mollahosseini et al. (2017)
- RAF-DB Dataset — Li et al. (2017)
- EfficientNet — Tan & Le (2019)
- YOLOv8 — Ultralytics (2023)
- OpenRouter API — https://openrouter.ai

---

## Common Issues

| Issue | Fix |
|-------|-----|
| Webcam not found | Run `usbipd attach --wsl --busid <id>` in PowerShell Admin |
| API 401 error | Check `.env` file has correct OpenRouter API key |
| API 402 error | Add credits at openrouter.ai/settings/credits |
| Qt platform error | Run `export DISPLAY=:0` before launching |
| CUDA not available | Reinstall PyTorch with `--index-url https://download.pytorch.org/whl/cu121` |

---

## License

University of Auckland — CS731 Human Robot Interaction Project 2026
