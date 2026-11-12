"""
utils/image_processor.py
------------------------
All image I/O, preprocessing, and display helpers.
Used by predict.py and app.py so that model-serving code is never
mixed with preprocessing logic.
"""

import io
from pathlib import Path
from typing import Union, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from config import IMAGE_SIZE, IMAGE_CHANNELS
from utils import get_logger

logger = get_logger(__name__)

# Type alias for accepted image inputs
ImageInput = Union[str, Path, bytes, Image.Image, np.ndarray]


# ─────────────────────────────────────────────
#  Loading
# ─────────────────────────────────────────────

def load_image(source: ImageInput) -> Image.Image:
    """
    Load an image from a file path, bytes buffer, or PIL/numpy object.

    Parameters
    ----------
    source : str | Path | bytes | PIL.Image | np.ndarray

    Returns
    -------
    PIL.Image.Image  (mode RGB)

    Raises
    ------
    ValueError
        If the source type is unrecognised.
    UnidentifiedImageError
        If Pillow cannot parse the image.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        img = Image.open(path)

    elif isinstance(source, bytes):
        img = Image.open(io.BytesIO(source))

    elif isinstance(source, Image.Image):
        img = source

    elif isinstance(source, np.ndarray):
        img = Image.fromarray(np.uint8(source))

    else:
        raise ValueError(
            f"Unsupported source type '{type(source).__name__}'. "
            "Expected str, Path, bytes, PIL.Image, or np.ndarray."
        )

    # Ensure RGB (drop alpha channel or convert from grayscale)
    img = img.convert("RGB")
    logger.debug("Image loaded — original size: %s", img.size)
    return img


# ─────────────────────────────────────────────
#  Preprocessing
# ─────────────────────────────────────────────

def preprocess_image(
    source: ImageInput,
    target_size: Tuple[int, int] = IMAGE_SIZE,
    expand_dims: bool = True,
) -> np.ndarray:
    """
    Resize and normalise an image for inference.

    Steps
    -----
    1. Load → RGB PIL image.
    2. Resize to `target_size` using high-quality Lanczos resampling.
    3. Convert to float32 numpy array.
    4. Normalise pixel values to [0, 1].
    5. Optionally add a batch dimension (required by model.predict).

    Parameters
    ----------
    source : ImageInput
    target_size : (height, width)
    expand_dims : bool
        If True, shape becomes (1, H, W, C); else (H, W, C).

    Returns
    -------
    np.ndarray  dtype=float32
    """
    img = load_image(source)
    img = img.resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS)

    array = np.array(img, dtype=np.float32) / 255.0  # normalise to [0, 1]

    if array.shape[-1] != IMAGE_CHANNELS:
        raise ValueError(
            f"Expected {IMAGE_CHANNELS} channels, got {array.shape[-1]}."
        )

    if expand_dims:
        array = np.expand_dims(array, axis=0)  # (1, H, W, C)

    logger.debug("Preprocessed image shape: %s", array.shape)
    return array


def batch_preprocess(
    image_paths: list,
    target_size: Tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    """
    Preprocess a list of image paths into a single batch tensor.

    Parameters
    ----------
    image_paths : list[str | Path]
    target_size : (height, width)

    Returns
    -------
    np.ndarray  shape (N, H, W, C)
    """
    arrays = []
    for path in image_paths:
        try:
            arr = preprocess_image(path, target_size, expand_dims=False)
            arrays.append(arr)
        except Exception as exc:
            logger.warning("Skipping '%s': %s", path, exc)

    if not arrays:
        raise ValueError("No valid images found in the provided list.")

    batch = np.stack(arrays, axis=0)
    logger.info("Batch tensor shape: %s", batch.shape)
    return batch


# ─────────────────────────────────────────────
#  Display Utilities
# ─────────────────────────────────────────────

def display_prediction(
    image_source: ImageInput,
    class_name: str,
    confidence: float,
    save_path: Optional[str] = None,
) -> None:
    """
    Display (and optionally save) an image with its predicted label overlay.

    Parameters
    ----------
    image_source : ImageInput
    class_name : str
    confidence : float  (0–1)
    save_path : str | None
        If provided, the figure is saved there instead of shown interactively.
    """
    img = load_image(image_source)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    ax.axis("off")

    color = "green" if "healthy" in class_name.lower() else "red"
    title = f"{class_name}\nConfidence: {confidence * 100:.1f}%"
    ax.set_title(title, fontsize=14, fontweight="bold", color=color, pad=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        logger.info("Prediction image saved to: %s", save_path)
    else:
        plt.show()

    plt.close(fig)


def plot_training_history(history, save_path: Optional[str] = None) -> None:
    """
    Plot training and validation accuracy/loss curves.

    Parameters
    ----------
    history : keras.callbacks.History
        Object returned by model.fit().
    save_path : str | None
        Path to save the plot as a PNG.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Accuracy ──────────────────────────────
    ax1.plot(history.history["accuracy"], label="Train Accuracy", marker="o")
    ax1.plot(history.history["val_accuracy"], label="Val Accuracy", marker="s")
    ax1.set_title("Model Accuracy", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ── Loss ──────────────────────────────────
    ax2.plot(history.history["loss"], label="Train Loss", marker="o")
    ax2.plot(history.history["val_loss"], label="Val Loss", marker="s")
    ax2.set_title("Model Loss", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle("Training History", fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        logger.info("Training history plot saved: %s", save_path)
    else:
        plt.show()

    plt.close(fig)


def plot_confusion_matrix(
    cm_array: np.ndarray,
    class_names: list,
    save_path: Optional[str] = None,
) -> None:
    """
    Render and save a colour-coded confusion matrix.

    Parameters
    ----------
    cm_array : np.ndarray  shape (N, N)
    class_names : list[str]
    save_path : str | None
    """
    import itertools

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(8, n), max(6, n - 2)))

    im = ax.imshow(cm_array, interpolation="nearest", cmap=cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(n),
        yticks=np.arange(n),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted Label",
        ylabel="True Label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = cm_array.max() / 2.0
    for i, j in itertools.product(range(n), range(n)):
        ax.text(
            j, i, str(cm_array[i, j]),
            ha="center", va="center",
            color="white" if cm_array[i, j] > threshold else "black",
            fontsize=max(6, 10 - n // 5),
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        logger.info("Confusion matrix saved: %s", save_path)
    else:
        plt.show()

    plt.close(fig)


# ─────────────────────────────────────────────
#  Grad-CAM (explainability)
# ─────────────────────────────────────────────

def generate_gradcam(
    model,
    image_array: np.ndarray,
    last_conv_layer_name: str,
    pred_index: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a Gradient-weighted Class Activation Map (Grad-CAM) heatmap.

    Grad-CAM highlights the regions of the image that most influenced
    the model's prediction, aiding interpretability.

    Parameters
    ----------
    model : keras.Model
    image_array : np.ndarray  shape (1, H, W, C), values in [0, 1]
    last_conv_layer_name : str
        Name of the last convolutional layer in the model.
    pred_index : int | None
        Class index to explain; defaults to the argmax prediction.

    Returns
    -------
    np.ndarray  shape (H, W)  — heatmap values in [0, 1]
    """
    import tensorflow as tf

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def overlay_gradcam(
    original_image: ImageInput,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Superimpose a Grad-CAM heatmap on the original image.

    Parameters
    ----------
    original_image : ImageInput
    heatmap : np.ndarray  shape (H, W)
    alpha : float
        Blending factor (0 = only image, 1 = only heatmap).
    save_path : str | None

    Returns
    -------
    np.ndarray  — blended RGB image (uint8)
    """
    img = load_image(original_image)
    img_array = np.array(img)

    # Resize heatmap to match image
    heatmap_resized = np.array(
        Image.fromarray(np.uint8(heatmap * 255)).resize(
            (img_array.shape[1], img_array.shape[0]), Image.Resampling.LANCZOS
        )
    )

    # Apply colour map
    colormap = cm.get_cmap("jet")
    heatmap_colored = np.uint8(colormap(heatmap_resized / 255.0) * 255)[:, :, :3]

    blended = np.uint8(img_array * (1 - alpha) + heatmap_colored * alpha)

    if save_path:
        Image.fromarray(blended).save(save_path)
        logger.info("Grad-CAM overlay saved: %s", save_path)

    return blended

