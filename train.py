"""
train.py
--------
Entry point for model training.

Usage
-----
  python train.py                        # train with defaults from config.py
  python train.py --model MobileNetV2   # use transfer-learning backbone (default)
  python train.py --model EfficientNetB0
  python train.py --model custom        # build CNN from scratch
  python train.py --epochs 50           # override epoch count
  python train.py --resume              # resume from last checkpoint

Auto-detects flat (PlantVillage) vs pre-split (dataset/train/) layout.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    EPOCHS, SAVED_MODEL_PATH, TRAINING_HISTORY_PLOT,
    CHECKPOINT_PATH, BATCH_SIZE,
)
from utils import get_logger
from utils.data_loader import (
    validate_dataset_structure,
    use_flat_layout,
    build_flat_datasets,
    build_train_generator,
    build_val_generator,
    compute_class_weights,
    dataset_summary,
)
from utils.model_builder import (
    build_custom_cnn,
    build_transfer_model,
    build_callbacks,
    save_model,
    load_model,
)
from utils.image_processor import plot_training_history

logger = get_logger(__name__)


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Plant Disease Detection CNN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, default="MobileNetV2",
        choices=["MobileNetV2", "EfficientNetB0", "custom"],
        help="Model architecture to use.",
    )
    parser.add_argument("--epochs",     type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--resume",     action="store_true",
                        help="Resume from last checkpoint.")
    parser.add_argument("--no-class-weights", action="store_true",
                        help="Disable class-weight balancing.")
    return parser.parse_args()


# ─────────────────────────────────────────────
#  Main Training Routine
# ─────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    logger.info("=" * 60)
    logger.info("  Plant Disease Detection — Training")
    logger.info("  Model : %s | Epochs : %d | Batch : %d",
                args.model, args.epochs, args.batch_size)
    logger.info("=" * 60)

    # ── 1. Validate dataset ───────────────────────────────────────
    logger.info("Step 1/5 — Validating dataset …")
    if not validate_dataset_structure():
        logger.error("Dataset validation failed. Check config.py paths.")
        sys.exit(1)

    # ── 2. Build data pipelines ───────────────────────────────────
    logger.info("Step 2/5 — Building data pipelines …")

    if use_flat_layout():
        # ── FLAT layout (PlantVillage) ────────────────────────────
        train_ds, val_ds, test_ds, class_names = build_flat_datasets(
            batch_size=args.batch_size,
            augment_train=True,
        )
        num_classes  = len(class_names)
        steps_per_epoch  = train_ds.cardinality().numpy()
        validation_steps = val_ds.cardinality().numpy()
        dataset_summary(train_ds, "Train")
        dataset_summary(val_ds,   "Validation")

        # Class weights from flat directory
        class_weight = None
        if not args.no_class_weights:
            try:
                class_weight = compute_class_weights(class_names)
            except Exception as exc:
                logger.warning("Class weights skipped: %s", exc)

        fit_kwargs = dict(
            x=train_ds,
            validation_data=val_ds,
            steps_per_epoch=steps_per_epoch if steps_per_epoch > 0 else None,
            validation_steps=validation_steps if validation_steps > 0 else None,
        )

    else:
        # ── PRE-SPLIT layout (dataset/train/) ─────────────────────
        train_gen = build_train_generator(batch_size=args.batch_size)
        val_gen   = build_val_generator(batch_size=args.batch_size)
        num_classes  = train_gen.num_classes
        class_names  = list(train_gen.class_indices.keys())

        class_weight = None
        if not args.no_class_weights:
            try:
                from sklearn.utils.class_weight import compute_class_weight
                import numpy as np
                labels  = train_gen.classes
                classes = np.unique(labels)
                weights = compute_class_weight("balanced", classes=classes, y=labels)
                class_weight = dict(zip(classes.tolist(), weights.tolist()))
            except Exception as exc:
                logger.warning("Class weights skipped: %s", exc)

        fit_kwargs = dict(
            x=train_gen,
            validation_data=val_gen,
            steps_per_epoch=max(1, train_gen.samples // args.batch_size),
            validation_steps=max(1, val_gen.samples  // args.batch_size),
        )

    if num_classes < 2:
        logger.error("Need at least 2 classes. Found %d.", num_classes)
        sys.exit(1)

    logger.info("Number of classes : %d", num_classes)
    logger.info("Classes           : %s", class_names)

    # ── 3. Build or resume model ──────────────────────────────────
    logger.info("Step 3/5 — Building model …")
    if args.resume and os.path.exists(CHECKPOINT_PATH):
        logger.info("Resuming from checkpoint: %s", CHECKPOINT_PATH)
        model = load_model(CHECKPOINT_PATH)
    elif args.model == "custom":
        model = build_custom_cnn(num_classes)
    else:
        model = build_transfer_model(num_classes, base_model_name=args.model)

    # ── 4. Train ──────────────────────────────────────────────────
    logger.info("Step 4/5 — Training …")
    callbacks = build_callbacks(checkpoint_path=CHECKPOINT_PATH)

    history = model.fit(
        **fit_kwargs,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    # ── 5. Save + plot ────────────────────────────────────────────
    logger.info("Step 5/5 — Saving model and plots …")
    save_model(model, SAVED_MODEL_PATH)
    plot_training_history(history, save_path=TRAINING_HISTORY_PLOT)

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("Model saved  : %s", SAVED_MODEL_PATH)
    logger.info("History plot : %s", TRAINING_HISTORY_PLOT)
    logger.info("Next step    : python evaluate.py")
    logger.info("=" * 60)


# ─────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    try:
        train(args)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        logger.exception("Unexpected error during training: %s", exc)
        sys.exit(1)
