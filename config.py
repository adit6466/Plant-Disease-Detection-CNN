"""
config.py
---------
Central configuration file for the Plant Disease Detection System.
All hyperparameters, paths, and settings are defined here so that
other modules never hard-code magic numbers or directory strings.
"""

import os

# Fix matplotlib writable cache dir (must be set before importing matplotlib)
os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", ".matplotlib"),
)

# ─────────────────────────────────────────────
#  Base Paths
# ─────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR     = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR       = os.path.join(DATASET_DIR, "train")
VALID_DIR       = os.path.join(DATASET_DIR, "valid")
TEST_DIR        = os.path.join(DATASET_DIR, "test")

# ─────────────────────────────────────────────
#  Flat Dataset (PlantVillage — no pre-split)
# ─────────────────────────────────────────────
# The PlantVillage dataset ships as one flat folder of class sub-directories.
# When FLAT_DATASET_DIR exists and contains class folders, the data_loader
# will use Keras's built-in validation_split instead of expecting pre-split
# train/valid/test directories.  Original images are NEVER moved or copied.
FLAT_DATASET_DIR = os.path.join(BASE_DIR, "PlantVillage", "PlantVillage")
TRAIN_SPLIT      = 0.80   # fraction of data used for training
VALID_SPLIT      = 0.10   # fraction of data used for validation
TEST_SPLIT       = 0.10   # fraction of data used for final test evaluation
MODEL_DIR       = os.path.join(BASE_DIR, "models")
DATA_DIR        = os.path.join(BASE_DIR, "data")
SPLITS_DIR      = os.path.join(DATA_DIR, "splits")
OUTPUT_DIR      = os.path.join(BASE_DIR, "outputs")
GRAPHS_DIR      = os.path.join(OUTPUT_DIR, "graphs")
REPORTS_DIR     = os.path.join(OUTPUT_DIR, "reports")
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")
ASSETS_DIR      = os.path.join(BASE_DIR, "assets")

# ─────────────────────────────────────────────
#  Dataset / Image Settings
# ─────────────────────────────────────────────
IMAGE_SIZE      = (224, 224)        # (height, width) fed to the CNN
IMAGE_CHANNELS  = 3                 # RGB
BATCH_SIZE      = 8                 # memory-safe for 8GB RAM / RTX 2050 / CPU
SEED            = 42                # for reproducibility

# ─────────────────────────────────────────────
#  Class Names (PlantVillage 15 classes)
# ─────────────────────────────────────────────
CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]
NUM_CLASSES = len(CLASS_NAMES)

# ─────────────────────────────────────────────
#  Model Architecture
# ─────────────────────────────────────────────
MODEL_NAME          = "plant_disease_cnn_v3"
SAVED_MODEL_PATH    = os.path.join(MODEL_DIR, f"{MODEL_NAME}.keras")
CHECKPOINT_PATH     = os.path.join(MODEL_DIR, f"{MODEL_NAME}_best.keras")

# ─────────────────────────────────────────────
#  Training Hyperparameters
# ─────────────────────────────────────────────
EPOCHS              = 30
LEARNING_RATE       = 5e-5
DROPOUT_RATE        = 0.5
L2_REGULARIZATION   = 1e-4

# Fine-tuning (transfer learning) settings
FINE_TUNE           = True          # enabled for high-accuracy MobileNetV2 fine-tuning
FINE_TUNE_AT        = 115           # unfreeze layers from index 115 onward (blocks 14-16)

# Early stopping / reduce-LR patience
EARLY_STOP_PATIENCE = 7
REDUCE_LR_PATIENCE  = 3
REDUCE_LR_FACTOR    = 0.5
MIN_LR              = 1e-7

# ─────────────────────────────────────────────
#  Data Augmentation (training only)
# ─────────────────────────────────────────────
AUGMENTATION = {
    "rotation_range":       20,
    "width_shift_range":    0.15,
    "height_shift_range":   0.15,
    "shear_range":          0.1,
    "zoom_range":           0.2,
    "horizontal_flip":      True,
    "vertical_flip":        False,
    "brightness_range":     [0.8, 1.2],
    "fill_mode":            "nearest",
}

# ─────────────────────────────────────────────
#  Disease Information CSV
# ─────────────────────────────────────────────
DISEASE_INFO_CSV = os.path.join(DATA_DIR, "disease_info.csv")

# ─────────────────────────────────────────────
#  Output Artifact Names
# ─────────────────────────────────────────────
DATASET_AUDIT_REPORT    = os.path.join(REPORTS_DIR, "dataset_audit_report.txt")
TRAINING_HISTORY_PLOT   = os.path.join(GRAPHS_DIR, "training_history_v3.png")
CONFUSION_MATRIX_PLOT   = os.path.join(GRAPHS_DIR, "confusion_matrix_v3.png")
CLASSIFICATION_REPORT   = os.path.join(REPORTS_DIR, "classification_report_v3.txt")
EVALUATION_RESULTS_CSV  = os.path.join(REPORTS_DIR, "evaluation_results_v3.csv")

# ─────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────
LOG_LEVEL = "INFO"      # DEBUG | INFO | WARNING | ERROR

# ─────────────────────────────────────────────
#  Streamlit App Settings
# ─────────────────────────────────────────────
APP_TITLE           = "🌿 Plant Disease Detection System"
APP_DESCRIPTION     = (
    "Upload a leaf image to detect whether the plant is healthy or diseased. "
    "Powered by a Convolutional Neural Network trained on the PlantVillage dataset."
)
CONFIDENCE_THRESHOLD = 0.60         # predictions below this are marked 'Uncertain'

