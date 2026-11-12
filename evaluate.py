"""
evaluate.py
-----------
Post-training evaluation on the hold-out test set.

Usage
-----
  python evaluate.py
  python evaluate.py --model-path models/my_model.keras

Outputs
-------
  outputs/graphs/confusion_matrix.png
  outputs/reports/classification_report.txt
  outputs/reports/evaluation_results.csv
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from config import (
    SAVED_MODEL_PATH,
    CONFUSION_MATRIX_PLOT,
    CLASSIFICATION_REPORT,
    EVALUATION_RESULTS_CSV,
    BATCH_SIZE,
)
from utils import get_logger
from utils.data_loader import (
    use_flat_layout,
    build_flat_datasets,
    build_test_generator,
    dataset_summary,
)
from utils.model_builder import load_model
from utils.image_processor import plot_confusion_matrix

logger = get_logger(__name__)


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the trained Plant Disease Detection model on the test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=SAVED_MODEL_PATH,
        help="Path to the trained model file (.keras).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )
    return parser.parse_args()


# ─────────────────────────────────────────────
#  Evaluation Logic
# ─────────────────────────────────────────────

def evaluate(args: argparse.Namespace) -> None:
    """
    Load the model, run inference on the test set, and produce reports.

    Parameters
    ----------
    args : argparse.Namespace
    """
    logger.info("═" * 60)
    logger.info("  Plant Disease Detection — Evaluation")
    logger.info("  Model : %s", args.model_path)
    logger.info("═" * 60)

    # ── 1. Load model ─────────────────────────────────────────────
    logger.info("Step 1/4 — Loading model …")
    model = load_model(args.model_path)

    # ── 2. Build test pipeline & run inference ────────────────────
    if use_flat_layout():
        logger.info("Step 2/4 — Building flat test dataset …")
        _, _, test_ds, class_names = build_flat_datasets(
            batch_size=args.batch_size,
            augment_train=False,
        )
        dataset_summary(test_ds, "Test")

        logger.info("Step 3/4 — Running inference on test dataset …")
        y_prob_list = []
        y_true_list = []
        for batch_x, batch_y in test_ds:
            probs = model.predict(batch_x, verbose=0)
            y_prob_list.append(probs)
            y_true_list.append(np.argmax(batch_y.numpy(), axis=1))

        y_prob = np.vstack(y_prob_list)
        y_true = np.concatenate(y_true_list)
        y_pred = np.argmax(y_prob, axis=1)
    else:
        logger.info("Step 2/4 — Building test generator …")
        test_gen = build_test_generator(batch_size=args.batch_size)
        class_names = list(test_gen.class_indices.keys())

        logger.info("Step 3/4 — Running inference on %d test images …", test_gen.samples)
        test_gen.reset()
        steps = -(-test_gen.samples // args.batch_size)   # ceiling division
        y_prob = model.predict(test_gen, steps=steps, verbose=1)
        y_pred = np.argmax(y_prob, axis=1)[:test_gen.samples]
        y_true = test_gen.classes[:test_gen.samples]

    # ── 4. Compute & Save Metrics ─────────────────────────────────
    logger.info("Step 4/4 — Computing metrics …")

    all_labels = list(range(len(class_names)))

    acc      = accuracy_score(y_true, y_pred)
    prec_mac = precision_score(y_true, y_pred, labels=all_labels, average="macro", zero_division=0)
    rec_mac  = recall_score(y_true, y_pred, labels=all_labels, average="macro", zero_division=0)
    f1_mac   = f1_score(y_true, y_pred, labels=all_labels, average="macro", zero_division=0)
    f1_wt    = f1_score(y_true, y_pred, labels=all_labels, average="weighted", zero_division=0)

    logger.info("Test Accuracy        : %.4f (%.2f%%)", acc, acc * 100)
    logger.info("Macro Precision      : %.4f", prec_mac)
    logger.info("Macro Recall         : %.4f", rec_mac)
    logger.info("Macro F1-Score       : %.4f", f1_mac)
    logger.info("Weighted F1-Score    : %.4f", f1_wt)

    # Classification report (text)
    report = classification_report(
        y_true, y_pred, labels=all_labels, target_names=class_names, zero_division=0
    )
    _save_classification_report(report, acc, prec_mac, rec_mac, f1_mac, f1_wt)

    # Confusion matrix (image)
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    plot_confusion_matrix(cm, class_names, save_path=CONFUSION_MATRIX_PLOT)

    # Per-class CSV
    _save_evaluation_csv(y_true, y_pred, y_prob, class_names)

    logger.info("Evaluation complete.")
    logger.info("Confusion matrix : %s", CONFUSION_MATRIX_PLOT)
    logger.info("Classification report : %s", CLASSIFICATION_REPORT)
    logger.info("Per-class CSV : %s", EVALUATION_RESULTS_CSV)


# ─────────────────────────────────────────────
#  Save Helpers
# ─────────────────────────────────────────────

def _save_classification_report(
    report: str,
    accuracy: float,
    precision_macro: float,
    recall_macro: float,
    f1_macro: float,
    f1_weighted: float,
) -> None:
    """Write the sklearn classification report to a text file."""
    os.makedirs(os.path.dirname(CLASSIFICATION_REPORT), exist_ok=True)

    bar = "=" * 50
    header = (
        f"Plant Disease Detection — Classification Report\n"
        f"{bar}\n"
        f"Overall Accuracy      : {accuracy * 100:.2f}%\n"
        f"Macro Precision       : {precision_macro:.4f}\n"
        f"Macro Recall          : {recall_macro:.4f}\n"
        f"Macro F1-Score        : {f1_macro:.4f}\n"
        f"Weighted F1-Score     : {f1_weighted:.4f}\n"
        f"{bar}\n\n"
    )

    with open(CLASSIFICATION_REPORT, "w", encoding="utf-8") as fh:
        fh.write(header + report)

    logger.info("Classification report saved: %s", CLASSIFICATION_REPORT)


def _save_evaluation_csv(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list,
) -> None:
    """
    Save a per-sample evaluation CSV with true label, predicted label,
    and the confidence score for the top prediction.
    """
    os.makedirs(os.path.dirname(EVALUATION_RESULTS_CSV), exist_ok=True)

    top_confidence = y_prob.max(axis=1)[:len(y_true)]
    top_pred_class = [class_names[i] for i in y_pred]
    true_class     = [class_names[i] for i in y_true]
    correct        = y_true == y_pred

    df = pd.DataFrame({
        "true_label":      true_class,
        "predicted_label": top_pred_class,
        "confidence":      top_confidence,
        "correct":         correct,
    })
    df.to_csv(EVALUATION_RESULTS_CSV, index=False, encoding="utf-8")
    logger.info("Evaluation CSV saved: %s", EVALUATION_RESULTS_CSV)


# ─────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    try:
        evaluate(args)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Evaluation interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        logger.exception("Unexpected error during evaluation: %s", exc)
        sys.exit(1)

