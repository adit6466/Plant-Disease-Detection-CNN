# 🌿 Plant Disease Detection System Using CNN

A production-ready Deep Learning system that classifies **38 plant disease/healthy categories** from leaf images using a MobileNetV2-based Convolutional Neural Network. Built with TensorFlow/Keras and served via a Streamlit web app.

---

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Dataset Setup](#dataset-setup)
- [Training](#training)
- [Evaluation](#evaluation)
- [Prediction](#prediction)
- [Web App](#web-app)
- [Configuration](#configuration)
- [File Reference](#file-reference)

---

## ✨ Features

| Feature | Detail |
|---|---|
| **Architecture** | MobileNetV2 transfer learning (or custom CNN) |
| **Classes** | 38 plant disease / healthy categories |
| **Input** | 224 × 224 RGB leaf images |
| **Augmentation** | Rotation, zoom, flip, brightness, shift |
| **Explainability** | Grad-CAM heatmap overlays |
| **UI** | Streamlit web app with confidence bars and disease info |
| **Reports** | Confusion matrix, classification report, per-sample CSV |
| **Platform** | Windows 11 / Linux / macOS |

---

## 🗂️ Project Structure

```
Plant_Disease_Detection/
│
├── app.py               ← Streamlit web application
├── train.py             ← Model training entry point
├── evaluate.py          ← Evaluation on test set
├── predict.py           ← CLI inference tool
├── config.py            ← All hyperparameters and paths
├── requirements.txt     ← Python dependencies
├── README.md
│
├── models/              ← Saved .keras models go here
├── dataset/
│   ├── train/           ← Training images (class sub-folders)
│   ├── valid/           ← Validation images
│   └── test/            ← Test images
│
├── data/
│   └── disease_info.csv ← Disease metadata for 38 classes
│
├── utils/
│   ├── __init__.py      ← Package init + shared logger
│   ├── data_loader.py   ← Dataset generators and tf.data pipelines
│   ├── model_builder.py ← CNN/transfer model builder + callbacks
│   ├── image_processor.py ← Image I/O, preprocessing, Grad-CAM
│   └── disease_info.py  ← CSV loader and query helpers
│
├── outputs/
│   ├── graphs/          ← Training curves, confusion matrix PNG
│   ├── reports/         ← Classification report TXT, evaluation CSV
│   └── predictions/     ← Annotated inference images
│
└── assets/
    └── images/          ← Static images for the README / app
```

---

## 🚀 Quick Start

### 1 — Clone / Navigate to the project

```bash
cd Plant_Disease_Detection
```

### 2 — Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate    # Linux / macOS
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📂 Dataset Setup

This project is designed to work with the **PlantVillage dataset**, which contains 54,000+ images across 38 categories.

### Download options

| Source | Link |
|---|---|
| Kaggle (recommended) | https://www.kaggle.com/datasets/emmarex/plantdisease |
| Official paper data | https://github.com/spMohanty/PlantVillage-Dataset |

### Folder structure expected

After downloading, organise your images as follows:

```
dataset/
├── train/
│   ├── Apple___Apple_scab/
│   │   ├── image1.jpg
│   │   └── ...
│   ├── Apple___Black_rot/
│   └── ... (one sub-folder per class)
│
├── valid/
│   ├── Apple___Apple_scab/
│   └── ...
│
└── test/
    ├── Apple___Apple_scab/
    └── ...
```

A common split is **80 / 10 / 10** (train / valid / test).

---

## 🏋️ Training

```bash
# Train with MobileNetV2 (default, recommended)
python train.py

# Train with EfficientNetB0
python train.py --model EfficientNetB0

# Train a custom CNN from scratch
python train.py --model custom

# Override epochs and batch size
python train.py --epochs 50 --batch-size 16

# Resume from last checkpoint
python train.py --resume
```

Training artifacts saved to:
- `models/plant_disease_cnn.keras` — final model
- `models/plant_disease_cnn_best.keras` — best checkpoint
- `outputs/graphs/training_history.png` — accuracy/loss curves

---

## 📊 Evaluation

```bash
python evaluate.py
```

Outputs:
- `outputs/graphs/confusion_matrix.png`
- `outputs/reports/classification_report.txt`
- `outputs/reports/evaluation_results.csv`

---

## 🔍 Prediction (CLI)

```bash
# Single image
python predict.py --image path/to/leaf.jpg

# Single image + save annotated result
python predict.py --image leaf.jpg --save-results

# Single image + Grad-CAM heatmap
python predict.py --image leaf.jpg --gradcam

# Batch folder prediction
python predict.py --folder path/to/folder --save-results
```

---

## 🌐 Web App

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

App features:
- Upload any leaf image (JPG, PNG, BMP, WEBP)
- Confidence bar + top-K predictions table
- Full disease description + treatment advice
- Optional Grad-CAM heatmap overlay
- Download JSON prediction report

---

## ⚙️ Configuration

All settings live in **`config.py`**. Key options:

| Parameter | Default | Description |
|---|---|---|
| `IMAGE_SIZE` | `(224, 224)` | Input resolution |
| `BATCH_SIZE` | `32` | Training batch size |
| `EPOCHS` | `30` | Maximum epochs |
| `LEARNING_RATE` | `1e-4` | Adam learning rate |
| `DROPOUT_RATE` | `0.5` | Head dropout probability |
| `FINE_TUNE` | `True` | Unfreeze top backbone layers |
| `FINE_TUNE_AT` | `100` | Unfreeze from this layer index |
| `CONFIDENCE_THRESHOLD` | `0.60` | Below this → "Uncertain" |
| `EARLY_STOP_PATIENCE` | `7` | Epochs before early stopping |

---

## 📄 File Reference

| File | Purpose |
|---|---|
| `config.py` | Single source of truth for all settings |
| `train.py` | Orchestrates the training pipeline |
| `evaluate.py` | Runs evaluation and generates reports |
| `predict.py` | CLI tool for single/batch inference |
| `app.py` | Streamlit web application |
| `utils/__init__.py` | Package init + shared logger factory |
| `utils/data_loader.py` | ImageDataGenerator and tf.data pipelines |
| `utils/model_builder.py` | CNN builder, callbacks, save/load helpers |
| `utils/image_processor.py` | Image I/O, Grad-CAM, plotting utilities |
| `utils/disease_info.py` | CSV loader and disease metadata queries |
| `data/disease_info.csv` | 38-class disease information reference table |
| `requirements.txt` | Python package dependencies |

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **TensorFlow / Keras** — model training and inference
- **MobileNetV2** — pre-trained ImageNet backbone
- **NumPy / Pandas** — numerical computing and data handling
- **Pillow** — image loading and manipulation
- **Matplotlib** — training plots and visualisations
- **Scikit-learn** — evaluation metrics
- **Streamlit** — interactive web application

---

## 📜 License

This project is released under the MIT License.

