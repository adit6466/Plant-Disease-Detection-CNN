"""
utils/model_builder.py
----------------------
Defines, compiles, and loads CNN models.

Two construction strategies are provided:
  1. build_custom_cnn   – a small CNN built from scratch (good for quick tests).
  2. build_transfer_model – MobileNetV2-based transfer-learning model (recommended
                            for real plant-disease datasets like PlantVillage).

Both return a compiled tf.keras.Model ready for training.
"""

from typing import Optional, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0

from config import (
    IMAGE_SIZE, IMAGE_CHANNELS, DROPOUT_RATE, L2_REGULARIZATION,
    LEARNING_RATE, FINE_TUNE, FINE_TUNE_AT, SAVED_MODEL_PATH,
)
from utils import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
#  Custom CNN (from scratch)
# ─────────────────────────────────────────────

def build_custom_cnn(num_classes: int) -> keras.Model:
    """
    Build a lightweight custom CNN for plant disease classification.

    Architecture
    ------------
    Input → [Conv2D → BN → ReLU → MaxPool] × 3 → Flatten → Dense → Dropout → Output

    Parameters
    ----------
    num_classes : int
        Number of disease/healthy categories.

    Returns
    -------
    keras.Model
        Compiled model.
    """
    input_shape = (*IMAGE_SIZE, IMAGE_CHANNELS)

    model = keras.Sequential([
        keras.Input(shape=input_shape, name="input"),

        # ── Block 1 ──────────────────────────────────
        layers.Conv2D(32, (3, 3), padding="same",
                      kernel_regularizer=regularizers.l2(L2_REGULARIZATION),
                      name="conv1"),
        layers.BatchNormalization(name="bn1"),
        layers.Activation("relu", name="relu1"),
        layers.MaxPooling2D((2, 2), name="pool1"),

        # ── Block 2 ──────────────────────────────────
        layers.Conv2D(64, (3, 3), padding="same",
                      kernel_regularizer=regularizers.l2(L2_REGULARIZATION),
                      name="conv2"),
        layers.BatchNormalization(name="bn2"),
        layers.Activation("relu", name="relu2"),
        layers.MaxPooling2D((2, 2), name="pool2"),

        # ── Block 3 ──────────────────────────────────
        layers.Conv2D(128, (3, 3), padding="same",
                      kernel_regularizer=regularizers.l2(L2_REGULARIZATION),
                      name="conv3"),
        layers.BatchNormalization(name="bn3"),
        layers.Activation("relu", name="relu3"),
        layers.MaxPooling2D((2, 2), name="pool3"),

        # ── Block 4 ──────────────────────────────────
        layers.Conv2D(256, (3, 3), padding="same",
                      kernel_regularizer=regularizers.l2(L2_REGULARIZATION),
                      name="conv4"),
        layers.BatchNormalization(name="bn4"),
        layers.Activation("relu", name="relu4"),
        layers.MaxPooling2D((2, 2), name="pool4"),

        # ── Classifier Head ───────────────────────────
        layers.GlobalAveragePooling2D(name="gap"),
        layers.Dense(512, activation="relu",
                     kernel_regularizer=regularizers.l2(L2_REGULARIZATION),
                     name="fc1"),
        layers.Dropout(DROPOUT_RATE, name="dropout"),
        layers.Dense(num_classes, activation="softmax", name="output"),
    ], name="PlantDiseaseCNN")

    model = _compile_model(model)
    logger.info("Custom CNN built with %d output classes.", num_classes)
    model.summary(print_fn=logger.info)
    return model


# ─────────────────────────────────────────────
#  Transfer-Learning Model (MobileNetV2)
# ─────────────────────────────────────────────

def build_transfer_model(
    num_classes: int,
    base_model_name: str = "MobileNetV2",
) -> keras.Model:
    """
    Build a transfer-learning model using a pre-trained backbone.

    Strategy
    --------
    1. Load a backbone pre-trained on ImageNet with the top removed.
    2. Freeze all backbone layers initially.
    3. Add a custom classification head.
    4. Optionally unfreeze the top `FINE_TUNE_AT` layers for fine-tuning
       (controlled by config.FINE_TUNE and config.FINE_TUNE_AT).

    Parameters
    ----------
    num_classes : int
    base_model_name : str
        "MobileNetV2" (default) or "EfficientNetB0".

    Returns
    -------
    keras.Model
        Compiled model.
    """
    input_shape = (*IMAGE_SIZE, IMAGE_CHANNELS)
    inputs = keras.Input(shape=input_shape, name="input")

    # ── Backbone ─────────────────────────────────────
    backbone = _get_backbone(base_model_name, input_shape)
    backbone.trainable = False          # freeze for feature extraction phase

    if FINE_TUNE:
        # Unfreeze layers from FINE_TUNE_AT onward
        for layer in backbone.layers[:FINE_TUNE_AT]:
            layer.trainable = False
        for layer in backbone.layers[FINE_TUNE_AT:]:
            layer.trainable = True
        logger.info(
            "Fine-tuning enabled: unfreezing layers from index %d.", FINE_TUNE_AT
        )

    x = backbone(inputs, training=False)

    # ── Classification Head ───────────────────────────
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(
        512, activation="relu",
        kernel_regularizer=regularizers.l2(L2_REGULARIZATION),
        name="fc1",
    )(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(DROPOUT_RATE, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs, outputs, name=f"PlantDisease_{base_model_name}")
    model = _compile_model(model)

    trainable_count = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    logger.info(
        "Transfer model (%s) built | %d output classes | %d trainable params.",
        base_model_name, num_classes, trainable_count,
    )
    return model


def _get_backbone(name: str, input_shape: Tuple) -> keras.Model:
    """
    Instantiate a pre-trained backbone by name.

    Parameters
    ----------
    name : str
    input_shape : Tuple

    Returns
    -------
    keras.Model
    """
    common_kwargs = dict(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    backbones = {
        "MobileNetV2":   lambda: MobileNetV2(**common_kwargs),
        "EfficientNetB0": lambda: EfficientNetB0(**common_kwargs),
    }
    if name not in backbones:
        raise ValueError(
            f"Unsupported backbone '{name}'. Choose from: {list(backbones.keys())}"
        )
    return backbones[name]()


# ─────────────────────────────────────────────
#  Compilation Helper
# ─────────────────────────────────────────────

def _compile_model(model: keras.Model) -> keras.Model:
    """
    Attach optimizer, loss, and metrics to the model.

    Uses Adam with the learning rate from config.py.

    Parameters
    ----------
    model : keras.Model

    Returns
    -------
    keras.Model (compiled)
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_accuracy"),
        ],
    )
    return model


# ─────────────────────────────────────────────
#  Callbacks Factory
# ─────────────────────────────────────────────

def build_callbacks(checkpoint_path: Optional[str] = None) -> list:
    """
    Return a list of standard training callbacks.

    Callbacks
    ---------
    - ModelCheckpoint  : saves the best model by val_accuracy.
    - EarlyStopping    : stops training when val_loss stops improving.
    - ReduceLROnPlateau: lowers the learning rate when stuck.
    - TensorBoard      : logs metrics for TensorBoard visualisation.

    Parameters
    ----------
    checkpoint_path : str | None
        Where to save the best checkpoint.  Defaults to config.CHECKPOINT_PATH.

    Returns
    -------
    list[keras.callbacks.Callback]
    """
    from config import (
        CHECKPOINT_PATH as _CKPT,
        EARLY_STOP_PATIENCE, REDUCE_LR_PATIENCE,
        REDUCE_LR_FACTOR, MIN_LR, OUTPUT_DIR,
    )

    ckpt = checkpoint_path or _CKPT

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=ckpt,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=MIN_LR,
            verbose=1,
        ),
    ]

    # TensorBoard is optional — only add if the package is installed
    try:
        import tensorboard  # noqa: F401
        tb_log_dir = str(OUTPUT_DIR + "/tensorboard_logs")
        callbacks.append(
            keras.callbacks.TensorBoard(
                log_dir=tb_log_dir,
                histogram_freq=1,
            )
        )
        logger.info("Callbacks configured: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard.")
    except ImportError:
        logger.warning("TensorBoard not installed — skipping TensorBoard callback. "
                       "Run: pip install tensorboard")
        logger.info("Callbacks configured: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau.")
    return callbacks


# ─────────────────────────────────────────────
#  Model Persistence
# ─────────────────────────────────────────────

def save_model(model: keras.Model, path: Optional[str] = None) -> None:
    """
    Save a trained model to disk in the native Keras format.

    Parameters
    ----------
    model : keras.Model
    path : str | None
        Defaults to config.SAVED_MODEL_PATH.
    """
    import os
    save_path = path or SAVED_MODEL_PATH
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    logger.info("Model saved to: %s", save_path)


def load_model(path: Optional[str] = None) -> keras.Model:
    """
    Load a saved Keras model from disk.

    Parameters
    ----------
    path : str | None
        Defaults to config.SAVED_MODEL_PATH.

    Returns
    -------
    keras.Model

    Raises
    ------
    FileNotFoundError
        If the model file does not exist.
    """
    import os
    load_path = path or SAVED_MODEL_PATH

    if not os.path.exists(load_path):
        raise FileNotFoundError(
            f"No model found at '{load_path}'. "
            "Train the model first by running: python train.py"
        )

    model = keras.models.load_model(load_path)
    logger.info("Model loaded from: %s", load_path)
    return model

