"""
evaluate_v3.py
--------------
Evaluates the fine-tuned v3 model on the untouched 2,064 holdout test images
from data/splits/test.csv and outputs comprehensive classification reports,
confusion matrix, and comparative metrics against the v2 baseline.
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import tensorflow as tf

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config import IMAGE_SIZE, IMAGE_CHANNELS, MODEL_DIR, GRAPHS_DIR, REPORTS_DIR, SPLITS_DIR
from utils import get_logger
from utils.data_loader import _read_split_csv, get_class_names
from utils.image_processor import plot_confusion_matrix

logger = get_logger("evaluate_v3")

MODEL_PATH = os.path.join(MODEL_DIR, "plant_disease_cnn_v3_best.keras")
CONFUSION_MATRIX_PATH = os.path.join(GRAPHS_DIR, "confusion_matrix_v3.png")
REPORT_PATH = os.path.join(REPORTS_DIR, "classification_report_v3.txt")
CSV_PATH = os.path.join(REPORTS_DIR, "evaluation_results_v3.csv")


def _decode_and_preprocess(filepath: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    raw = tf.io.read_file(filepath)
    img = tf.io.decode_jpeg(raw, channels=IMAGE_CHANNELS)
    img = tf.image.resize(img, IMAGE_SIZE, method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


def main():
    logger.info("=" * 65)
    logger.info("  Plant Disease Detection — Holdout Test Set Evaluation (v3)")
    logger.info("=" * 65)

    test_csv = Path(SPLITS_DIR) / "test.csv"
    if not test_csv.exists():
        logger.error("Test split manifest not found at %s", test_csv)
        sys.exit(1)

    test_rows = _read_split_csv(test_csv)
    class_names = get_class_names()
    num_classes = len(class_names)
    logger.info("Evaluating on %d holdout test images across %d classes.", len(test_rows), num_classes)

    fps = [r["filepath"] for r in test_rows]
    lbls = tf.keras.utils.to_categorical([r["class_index"] for r in test_rows], num_classes=num_classes)

    ds = tf.data.Dataset.from_tensor_slices((fps, lbls))
    ds = ds.map(_decode_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(64).prefetch(tf.data.AUTOTUNE)

    if not os.path.exists(MODEL_PATH):
        fallback = os.path.join(MODEL_DIR, "plant_disease_cnn_v3.keras")
        if os.path.exists(fallback):
            model_target = fallback
        else:
            logger.error("Neither %s nor %s found!", MODEL_PATH, fallback)
            sys.exit(1)
    else:
        model_target = MODEL_PATH

    logger.info("Loading model from %s...", model_target)
    model = tf.keras.models.load_model(model_target)

    logger.info("Running test inference...")
    t0 = time.time()
    y_prob_list = []
    y_true_list = []
    for batch_x, batch_y in ds:
        probs = model.predict(batch_x, verbose=0)
        y_prob_list.append(probs)
        y_true_list.append(np.argmax(batch_y.numpy(), axis=1))

    y_prob = np.vstack(y_prob_list)
    y_true = np.concatenate(y_true_list)
    y_pred = np.argmax(y_prob, axis=1)
    infer_time = time.time() - t0
    logger.info("Inference completed in %.2fs (%.2f ms/image).", infer_time, (infer_time / len(y_true)) * 1000)

    # Compute metrics
    all_labels = list(range(num_classes))
    acc = accuracy_score(y_true, y_pred)
    prec_mac = precision_score(y_true, y_pred, labels=all_labels, average="macro", zero_division=0)
    rec_mac = recall_score(y_true, y_pred, labels=all_labels, average="macro", zero_division=0)
    f1_mac = f1_score(y_true, y_pred, labels=all_labels, average="macro", zero_division=0)
    f1_wt = f1_score(y_true, y_pred, labels=all_labels, average="weighted", zero_division=0)

    logger.info("=" * 65)
    logger.info("  FINAL EVALUATION RESULTS (v3)")
    logger.info("  Overall Accuracy   : %.4f (%.2f%%)", acc, acc * 100)
    logger.info("  Macro Precision    : %.4f", prec_mac)
    logger.info("  Macro Recall       : %.4f", rec_mac)
    logger.info("  Macro F1-Score     : %.4f", f1_mac)
    logger.info("  Weighted F1-Score  : %.4f", f1_wt)
    logger.info("=" * 65)

    # Classification report
    rep_str = classification_report(
        y_true, y_pred, labels=all_labels, target_names=class_names, zero_division=0
    )

    header = (
        "Plant Disease Detection — Classification Report (v3 Fine-Tuned)\n"
        "=================================================================\n"
        f"Overall Accuracy      : {acc * 100:.2f}%\n"
        f"Macro Precision       : {prec_mac:.4f}\n"
        f"Macro Recall          : {rec_mac:.4f}\n"
        f"Macro F1-Score        : {f1_mac:.4f}\n"
        f"Weighted F1-Score     : {f1_wt:.4f}\n"
        f"Holdout Test Samples  : {len(y_true):,}\n"
        "=================================================================\n\n"
    )

    Path(REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(header + rep_str)
    logger.info("Saved classification report to %s", REPORT_PATH)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    plot_confusion_matrix(cm, class_names, save_path=CONFUSION_MATRIX_PATH)

    # Per-sample predictions CSV
    df = pd.DataFrame({
        "filepath": [r["filepath"] for r in test_rows],
        "true_label": [class_names[t] for t in y_true],
        "pred_label": [class_names[p] for p in y_pred],
        "confidence": [float(y_prob[i, y_pred[i]]) for i in range(len(y_true))],
        "correct": (y_true == y_pred),
    })
    df.to_csv(CSV_PATH, index=False)
    logger.info("Saved per-sample evaluation results to %s", CSV_PATH)


if __name__ == "__main__":
    main()
