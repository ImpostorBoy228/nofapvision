
# NoFapVision 🖤💀

AI classifier for recognizing two states: **fap** and **nofap** from video/webcam.

The project includes:
- Video frame extraction + augmentation (`build_dataset.py`)
- Neural network trainer with transfer learning (`train.py`)
- Real-time classification with webcam (`inference_camera.py`)
- PyTorch DataLoader and model (`dataset_loader.py` + `model.py`)

## 📁 Project Structure
```
nofapvision/
│
├─ dataset/          # Frames will be stored here
│   ├─ fap/
│   └─ nofap/
│
├─ build_dataset.py
├─ extract-frames.py
├─ augment_dataset.py
├─ dataset_loader.py
├─ model.py
├─ train.py
├─ inference_camera.py
├─ requirements.txt
└─ README.md
```

## ⚡ Installation
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows

pip install --upgrade pip
pip install torch torchvision opencv-python tqdm scikit-learn
```

## 🩸 Dataset Preparation
1. Place `fap.mp4` and `nofap.mp4` in the project root.
2. Run:
```bash
python build_dataset.py
```
- Frames will be extracted every **1 second**.
- Augmentations applied automatically.
- Output structure:
```
dataset/
  fap/
  nofap/
```

## 🧠 Model Training
```bash
python train.py --dataset ./dataset --epochs 15 --batch 32 --img_size 224
```
- MobileNetV2 transfer learning.
- Checkpoints stored in `checkpoints/`.
- Best model saved as `checkpoints/best_model.pt`.

## 🎥 Webcam Inference
```bash
python inference_camera.py --model ./checkpoints/best_model.pt --cam 0 --img_size 224
```
- Real-time video window.
- Displays prediction (`fap/nofap`) with probability.
- Quit with `q` key.

## 🛠️ Settings
- `FRAME_SIZE` in `build_dataset.py` — recommended **224x224** for MobileNet.
- Augmentations: flip, brightness, gamma, noise, combo effects.
- Frame interval (`INTERVAL_SEC`) — default 1 second.

## ⚠️ Notes
- Ensure both classes have enough frames.
- MobileNetV2 works on CPU but GPU speeds up training.
- To reduce prediction flicker, average probabilities over several frames (e.g., with `deque`).

