"""
predict.py
----------
Run inference on a single image or a folder of images.

Usage
-----
  python predict.py --image path/to/leaf.jpg
  python predict.py --folder path/to/folder --save-results
  python predict.py --image leaf.jpg --gradcam

Output
------
  - Predicted class name and confidence score (always printed to console).
  - Optionally saves an annotated image to outputs/predictions/.
  - Optionally generates a Grad-CAM overlay for visual explainability.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from config import (
    SAVED_MODEL_PATH,
    PREDICTIONS_DIR,
    CONFIDENCE_THRESHOLD,
    IMAGE_SIZE,
)
from utils import get_logger
from utils.model_builder import load_model
from utils.image_processor import (
    preprocess_image,
    batch_preprocess,
    display_prediction,
    generate_gradcam,
    overlay_gradcam,
)
from utils.disease_info import load_disease_info, get_disease_details

logger = get_logger(__name__)

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict plant disease from a leaf image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image",  type=str, help="Path to a single image file.")
    source.add_argument("--folder", type=str, help="Path to a folder of images.")

    parser.add_argument(
        "--model-path",
        type=str,
        default=SAVED_MODEL_PATH,
        help="Path to the trained model (.keras).",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save annotated prediction images to outputs/predictions/.",
    )
    parser.add_argument(
        "--gradcam",
        action="store_true",
        help="Generate Grad-CAM heatmap overlay (single image only).",
    )
    parser.add_argument(
        "--last-conv-layer",
        type=str,
        default=None,
        help="Name of the last conv layer for Grad-CAM (auto-detected if omitted).",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────
#  Core Prediction Logic
# ─────────────────────────────────────────────

def predict_single(
    image_path: str,
    model,
    class_names: list,
    disease_df,
    save: bool = False,
    gradcam: bool = False,
    last_conv_layer: Optional[str] = None,
) -> dict:
    """
    Classify a single leaf image.

    Parameters
    ----------
    image_path : str
    model : keras.Model
    class_names : list[str]
    disease_df : pd.DataFrame
    save : bool
        Whether to save the annotated output image.
    gradcam : bool
        Whether to produce a Grad-CAM overlay.
    last_conv_layer : str | None

    Returns
    -------
    dict  with keys: image_path, class_name, confidence, details
    """
    logger.info("Processing: %s", image_path)

    # Preprocess
    img_array = preprocess_image(image_path)   # shape (1, H, W, C)

    # Inference
    predictions = model.predict(img_array, verbose=0)   # (1, num_classes)
    pred_index  = int(np.argmax(predictions[0]))
    confidence  = float(predictions[0][pred_index])
    class_name  = class_names[pred_index]

    # Uncertainty threshold
    if confidence < CONFIDENCE_THRESHOLD:
        display_class = f"Uncertain ({class_name})"
        logger.warning(
            "Low confidence (%.2f%%) for '%s' — prediction marked Uncertain.",
            confidence * 100, class_name,
        )
    else:
        display_class = class_name

    # Disease metadata
    details = get_disease_details(class_name, disease_df)

    # Print result
    _print_result(image_path, display_class, confidence, details)

    # Optionally save annotated image
    if save:
        stem     = Path(image_path).stem
        out_path = os.path.join(PREDICTIONS_DIR, f"{stem}_prediction.png")
        os.makedirs(PREDICTIONS_DIR, exist_ok=True)
        display_prediction(image_path, display_class, confidence, save_path=out_path)

    # Optionally generate Grad-CAM
    if gradcam:
        conv_layer = last_conv_layer or _find_last_conv_layer(model)
        if conv_layer:
            heatmap = generate_gradcam(model, img_array, conv_layer, pred_index)
            stem     = Path(image_path).stem
            gc_path  = os.path.join(PREDICTIONS_DIR, f"{stem}_gradcam.png")
            os.makedirs(PREDICTIONS_DIR, exist_ok=True)
            overlay_gradcam(image_path, heatmap, save_path=gc_path)
            logger.info("Grad-CAM saved: %s", gc_path)
        else:
            logger.warning("No convolutional layer found — Grad-CAM skipped.")

    return {
        "image_path": image_path,
        "class_name": class_name,
        "confidence": confidence,
        "details":    details,
    }


def predict_folder(
    folder_path: str,
    model,
    class_names: list,
    disease_df,
    save: bool = False,
) -> list:
    """
    Run batch prediction on all images in a folder.

    Parameters
    ----------
    folder_path : str
    model : keras.Model
    class_names : list[str]
    disease_df : pd.DataFrame
    save : bool

    Returns
    -------
    list[dict]  One dict per image.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise NotADirectoryError(f"'{folder_path}' is not a valid directory.")

    image_paths = [
        str(p) for p in folder.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not image_paths:
        logger.warning("No supported images found in '%s'.", folder_path)
        return []

    logger.info("Found %d image(s) in '%s'.", len(image_paths), folder_path)

    results = []
    for path in sorted(image_paths):
        try:
            result = predict_single(
                path, model, class_names, disease_df, save=save, gradcam=False
            )
            results.append(result)
        except Exception as exc:
            logger.error("Failed to process '%s': %s", path, exc)

    _print_summary(results)
    return results


# ─────────────────────────────────────────────
#  Display Helpers
# ─────────────────────────────────────────────

def _print_result(image_path: str, class_name: str, confidence: float, details: dict) -> None:
    """Pretty-print a single prediction result."""
    sep = "─" * 55
    print(f"\n{sep}")
    print(f"  Image        : {os.path.basename(image_path)}")
    print(f"  Prediction   : {class_name}")
    print(f"  Confidence   : {confidence * 100:.2f}%")
    print(f"  Plant        : {details.get('plant_species', 'N/A')}")
    print(f"  Disease      : {details.get('disease_name',  'N/A')}")
    print(f"  Severity     : {details.get('severity',      'N/A')}")
    print(f"  Description  : {details.get('description',   'N/A')}")
    print(f"  Treatment    : {details.get('treatment',      'N/A')}")
    print(f"{sep}\n")


def _print_summary(results: list) -> None:
    """Print a tabular summary after batch inference."""
    print("\n" + "═" * 70)
    print(f"  BATCH PREDICTION SUMMARY — {len(results)} image(s)")
    print("═" * 70)
    for r in results:
        flag = "✓" if r["confidence"] >= CONFIDENCE_THRESHOLD else "?"
        print(
            f"  [{flag}] {os.path.basename(r['image_path']):<30} "
            f"{r['class_name']:<35} {r['confidence'] * 100:.1f}%"
        )
    print("═" * 70 + "\n")


def _find_last_conv_layer(model) -> Optional[str]:
    """
    Auto-detect the name of the last Conv2D layer in the model.
    Used as the default target for Grad-CAM.
    """
    for layer in reversed(model.layers):
        if hasattr(layer, "filters") or "conv" in layer.name.lower():
            logger.info("Auto-detected last conv layer: %s", layer.name)
            return layer.name
    return None


# ─────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    try:
        # Load model once
        model = load_model(args.model_path)

        # Derive class names from the model's output layer
        num_classes = model.output_shape[-1]
        logger.info("Model has %d output classes.", num_classes)

        # Attempt to read class names from train directory
        try:
            from utils.data_loader import get_class_names
            class_names = get_class_names()
            if len(class_names) != num_classes:
                raise ValueError(
                    f"Class name count ({len(class_names)}) does not match "
                    f"model output size ({num_classes})."
                )
        except Exception as exc:
            logger.warning(
                "Could not read class names from dataset: %s. "
                "Using generic labels (class_0, class_1, …).",
                exc,
            )
            class_names = [f"class_{i}" for i in range(num_classes)]

        # Load disease info (may be empty if CSV is absent)
        disease_df = load_disease_info()

        # Run prediction
        if args.image:
            predict_single(
                args.image,
                model,
                class_names,
                disease_df,
                save=args.save_results,
                gradcam=args.gradcam,
                last_conv_layer=args.last_conv_layer,
            )
        else:
            predict_folder(
                args.folder,
                model,
                class_names,
                disease_df,
                save=args.save_results,
            )

    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Prediction interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        sys.exit(1)

