"""
train_v3.py
-----------
Fine-tunes MobileNetV2 for the Plant Disease Detection System (v3).
Unfreezes top convolutional blocks (layers 115+) of MobileNetV2 with frozen
BatchNormalization layers, using balanced class weights and a fine-tuning
learning rate (5e-5) to optimize classification accuracy towards ~94%.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config import (
    IMAGE_SIZE, IMAGE_CHANNELS, SEED, SPLITS_DIR, MODEL_DIR,
    GRAPHS_DIR, REPORTS_DIR, AUGMENTATION
)
from utils import get_logger
from utils.data_loader import _read_split_csv, compute_class_weights, get_class_names

logger = get_logger("train_v3")

BATCH_SIZE = 64
EPOCHS = 6
INITIAL_LR = 5e-5
V2_MODEL_PATH = os.path.join(MODEL_DIR, "plant_disease_cnn_v2_best.keras")
V3_BEST_MODEL_PATH = os.path.join(MODEL_DIR, "plant_disease_cnn_v3_best.keras")
V3_FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "plant_disease_cnn_v3.keras")
V3_HISTORY_PLOT = os.path.join(GRAPHS_DIR, "training_history_v3.png")


def _decode_and_preprocess(filepath: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Fast JPEG decode and normalization."""
    raw = tf.io.read_file(filepath)
    img = tf.io.decode_jpeg(raw, channels=IMAGE_CHANNELS)
    img = tf.image.resize(img, IMAGE_SIZE, method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


def _build_augmentation() -> tf.keras.Sequential:
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal", seed=SEED),
        layers.RandomRotation(AUGMENTATION["rotation_range"] / 360, seed=SEED),
        layers.RandomZoom(AUGMENTATION["zoom_range"], seed=SEED),
        layers.RandomTranslation(
            AUGMENTATION["height_shift_range"],
            AUGMENTATION["width_shift_range"],
            seed=SEED,
        ),
        layers.RandomBrightness(
            factor=(
                AUGMENTATION["brightness_range"][0] - 1.0,
                AUGMENTATION["brightness_range"][1] - 1.0,
            ),
            value_range=(0.0, 1.0),
            seed=SEED,
        ),
    ], name="data_augmentation")


def create_pipeline(rows: List[dict], num_classes: int, is_training: bool = True) -> tf.data.Dataset:
    fps = [r["filepath"] for r in rows]
    lbls = tf.keras.utils.to_categorical([r["class_index"] for r in rows], num_classes=num_classes)

    ds = tf.data.Dataset.from_tensor_slices((fps, lbls))
    if is_training:
        ds = ds.shuffle(buffer_size=min(len(rows), 2048), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(_decode_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)

    if is_training:
        aug_layer = _build_augmentation()
        ds = ds.map(lambda x, y: (aug_layer(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def plot_and_save_history(history, save_path: str):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    epochs_range = range(1, len(history.history["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax1.plot(epochs_range, history.history["accuracy"], "o-", label="Train Accuracy", color="#176B50")
    ax1.plot(epochs_range, history.history["val_accuracy"], "s--", label="Val Accuracy", color="#2DD4A0")
    ax1.set_title("Training & Validation Accuracy (v3 Fine-Tuning)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right")

    # Loss
    ax1_loss = ax2
    ax1_loss.plot(epochs_range, history.history["loss"], "o-", label="Train Loss", color="#D6B56D")
    ax1_loss.plot(epochs_range, history.history["val_loss"], "s--", label="Val Loss", color="#C0392B")
    ax1_loss.set_title("Training & Validation Loss (v3 Fine-Tuning)", fontsize=12, fontweight="bold")
    ax1_loss.set_xlabel("Epoch")
    ax1_loss.set_ylabel("Loss")
    ax1_loss.grid(True, linestyle="--", alpha=0.5)
    ax1_loss.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    logger.info("Saved training history plot to %s", save_path)


def main():
    logger.info("=" * 65)
    logger.info("  Plant Disease Detection — Phase 2 Fine-Tuning (v3)")
    logger.info("=" * 65)

    splits_path = Path(SPLITS_DIR)
    train_csv = splits_path / "train.csv"
    valid_csv = splits_path / "valid.csv"
    test_csv = splits_path / "test.csv"

    if not (train_csv.exists() and valid_csv.exists() and test_csv.exists()):
        logger.error("Splits manifests not found in %s", splits_path)
        sys.exit(1)

    train_rows = _read_split_csv(train_csv)
    valid_rows = _read_split_csv(valid_csv)
    class_names = get_class_names()
    num_classes = len(class_names)

    logger.info("Train set: %d images | Val set: %d images | Classes: %d",
                len(train_rows), len(valid_rows), num_classes)

    class_weights = compute_class_weights(class_names)
    logger.info("Computed balanced class weights for %d classes.", len(class_weights))

    train_ds = create_pipeline(train_rows, num_classes, is_training=True)
    val_ds = create_pipeline(valid_rows, num_classes, is_training=False)

    logger.info("Loading pre-trained base model from: %s", V2_MODEL_PATH)
    model = tf.keras.models.load_model(V2_MODEL_PATH)

    # Find MobileNetV2 backbone
    backbone = model.get_layer("mobilenetv2_1.00_224")
    backbone.trainable = True

    # Unfreeze layers from index 115 onward, keeping all BatchNorm layers frozen
    UNFREEZE_FROM = 115
    unfrozen_count = 0
    bn_frozen_count = 0

    for i, layer in enumerate(backbone.layers):
        if i < UNFREEZE_FROM:
            layer.trainable = False
        else:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
                bn_frozen_count += 1
            else:
                layer.trainable = True
                unfrozen_count += 1

    logger.info(
        "MobileNetV2 unfreezing: layers %d to %d unfrozen (%d conv/dense layers active, %d BN frozen).",
        UNFREEZE_FROM, len(backbone.layers) - 1, unfrozen_count, bn_frozen_count
    )

    optimizer = keras.optimizers.Adam(learning_rate=INITIAL_LR)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_accuracy")],
    )

    trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    non_trainable_params = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)
    logger.info("Total Trainable Parameters: %d | Non-Trainable: %d", trainable_params, non_trainable_params)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=V3_BEST_MODEL_PATH,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    logger.info("Starting fine-tuning for %d epochs at learning_rate=%e with batch_size=%d...",
                EPOCHS, INITIAL_LR, BATCH_SIZE)
    t_start = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )
    t_elapsed = time.time() - t_start
    logger.info("Fine-tuning completed in %.2f minutes (%.1f seconds).", t_elapsed / 60, t_elapsed)

    # Save final model
    model.save(V3_FINAL_MODEL_PATH)
    logger.info("Saved final model to %s", V3_FINAL_MODEL_PATH)

    # Plot history
    plot_and_save_history(history, V3_HISTORY_PLOT)

    logger.info("Model v3 fine-tuning successfully finished!")


if __name__ == "__main__":
    main()
