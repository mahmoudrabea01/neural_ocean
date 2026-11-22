# Neural Ocean - YOLOv8s Trash Detection Model

This repository contains the training runs, configuration settings, and optimized weights for detecting marine debris and underwater trash using the Ultralytics YOLOv8 framework.

the other CV tasks
object_count: https://github.com/mahmoudrabea01/object_count
fire_and_smoke_detecation: https://github.com/mahmoudrabea01/fire_smoke_detection
eye_vision: https://github.com/mahmoudrabea01/eye_vision_task
---

## 🚀 How to Clone or Pull this Repository

### 1. Clone the Repository
To download the full repository including all model weights and performance curves, run the following command in your local terminal:
```bash
git clone https://github.com/mahmoudrabea01/neural_ocean.git
cd neural_ocean
```

### 2. Pull Latest Updates
If you have already cloned the repository and want to sync with the latest remote modifications:
```bash
git pull origin main
```

---

## 📝 Notebook Implementation Overview
In this project, we executed a complete end-to-end Machine Learning pipeline inside Google Colab:
1. **Environment Initialization:** Configured PyTorch and verified CUDA execution on an NVIDIA Tesla T4 GPU.
2. **Dataset Retrieval:** Authenticated with Roboflow using an API Key to fetch version 3 of the `neural_ocean` dataset in the YOLO format.
3. **Model Selection:** Initialized a lightweight, highly accurate standard pre-trained checkpoint `yolo26s.pt` from Ultralytics.
4. **Custom Model Training:** Fine-tuned the detector for **60 epochs** using custom parameters tuned for marine objects.
5. **Validation & Benchmark:** Evaluated training performance using precision/recall metrics, and validated generalization on a standalone test split.

---

## 📊 Dataset Breakdown

### What is the dataset used for?
The `neural_ocean` dataset is designed for marine ecology preservation. It helps researchers, automated underwater vehicles (AUVs), and cleanup devices autonomously detect, classify, and track pollution, litter, and equipment underwater.

### Class Names (15 Classes in Total)
* `Mask` | `can` | `cellphone` | `electronics` | `gbottle` (Glass bottle)
* `glove` | `metal` | `misc` (Miscellaneous) | `net` (Fishing net) | `pbag` (Plastic bag)
* `pbottle` (Plastic bottle) | `plastic` | `rod` | `sunglasses` | `tire`

### Dataset Directory Organization
After downloading version 3 of the dataset, it is laid out in standard YOLO format:
```
/content/Neural_Ocean-3/
├── data.yaml            # Paths to train/val/test splits and class names
├── train/
│   ├── images/          # Images used for training
│   └── labels/          # Text files with normalized bounding box annotations
├── val/
│   ├── images/          # Validation images used to measure performance per epoch
│   └── labels/          # Validation ground-truth annotations
└── test/
    ├── images/          # Test set images for final generalization benchmarks
    └── labels/          # Ground-truth annotations for test set evaluation
```

---

## ⚙️ Understanding the Training Parameters

### 1. Image Size (`imgsz=640`)
* **Impact:** Resizes the input images to 640x640 pixels during training and evaluation.
* **Why it matters:** Larger image sizes allow the model to capture smaller, finer details of debris (e.g., small plastic pieces further away), but they demand significantly more GPU VRAM and increase inference time. `640` is the industry-standard sweet spot for speed and precision.

### 2. Patience (`patience=10`)
* **Impact:** Early Stopping patience parameter.
* **Why it matters:** If the validation loss does not show improvement for 10 consecutive epochs, training terminates early. This prevents wasting computing resources and mitigates overfitting, ensuring that the model retains its optimal generalized form.

### 3. Optimizer & Learning Rate
* **Optimizer:** Autoselected (such as AdamW or SGD) to calculate and apply gradients to update model parameters.
* **Learning Rate (LR):** Controls step sizes towards the loss function minimum. A learning rate that is too high might cause training to diverge, while a rate that is too low might result in incredibly slow convergence or getting stuck in local minima.

---

## 🏋️ Weights, Inference, & Deployment

### Which file is used for final prediction?
The `best.pt` weight file is located under `yolo26s/weights/best.pt`. This file must be used for final model inferences because it represents the checkpoint that achieved the highest mAP (mean Average Precision) on the validation set during the training epochs.

### Why save model weights and how to reuse them?
Saving model weights preserves the parameters (biases and learned kernel shapes) optimized during training. It avoids wasting time and computational cost on retraining. To reuse these weights later in another application or notebook, load them directly with Python:
```python
from ultralytics import YOLO

# Instantiate and load trained weights
model = YOLO("path/to/best.pt")

# Run model inference on a new image or video
results = model.predict(source="ocean_trash.jpg", conf=0.25, save=True)
```

### Which file is better for deployment: `best.pt` or `last.pt`?
* **`best.pt` is almost always preferred for deployment.** It represents the peak-performing state of the network during training, offering the highest accuracy on unseen datasets.
* **`last.pt`** is simply the state of the model at the very last epoch. If training stops early or overfits in later stages, `last.pt` might actually exhibit degraded performance or high validation loss. It is primarily kept to resume interrupted training runs.

