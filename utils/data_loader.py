"""
utils/data_loader.py
--------------------
Responsible for loading, stratified splitting, deduplicating, and preparing
image datasets without moving or copying original images.

Supports TWO dataset layouts automatically:

  Layout A — FLAT (PlantVillage style, recommended):
      PlantVillage/PlantVillage/<ClassName>/*.jpg
      Uses deterministic, SHA256-aware, stratified 80/10/10 split manifests
      stored in data/splits/{train,valid,test}.csv and streams images via
      tf.data with prefetch(1) and NO .cache() (safe for 8GB RAM).

  Layout B — PRE-SPLIT (manual split):
      dataset/train/<ClassName>/
      dataset/valid/<ClassName>/
      dataset/test/<ClassName>/
      Uses ImageDataGenerator.flow_from_directory.
"""

import csv
import hashlib
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import (
    AUGMENTATION,
    BATCH_SIZE,
    DATASET_AUDIT_REPORT,
    FLAT_DATASET_DIR,
    IMAGE_CHANNELS,
    IMAGE_SIZE,
    SEED,
    SPLITS_DIR,
    TEST_DIR,
    TEST_SPLIT,
    TRAIN_DIR,
    TRAIN_SPLIT,
    VALID_DIR,
    VALID_SPLIT,
)
from utils import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ─────────────────────────────────────────────
#  Layout Detection
# ─────────────────────────────────────────────

def _has_images(directory: str) -> bool:
    """Return True if `directory` contains at least one supported image file."""
    p = Path(directory)
    if not p.is_dir():
        return False
    for sub in p.iterdir():
        if sub.is_dir():
            for f in sub.iterdir():
                if f.suffix.lower() in SUPPORTED_EXTS:
                    return True
    return False


def use_flat_layout() -> bool:
    """
    Return True when the flat PlantVillage folder exists and contains images.
    Falls back to the pre-split layout (dataset/train/) otherwise.
    """
    flat_ok = _has_images(FLAT_DATASET_DIR)
    split_ok = _has_images(TRAIN_DIR)
    if flat_ok:
        logger.info("Dataset layout : FLAT  -> %s", FLAT_DATASET_DIR)
    elif split_ok:
        logger.info("Dataset layout : PRE-SPLIT  -> %s", TRAIN_DIR)
    else:
        logger.error(
            "No dataset found.\n"
            "  Flat path checked  : %s\n"
            "  Split path checked : %s",
            FLAT_DATASET_DIR,
            TRAIN_DIR,
        )
    return flat_ok


# ─────────────────────────────────────────────
#  Dataset Validation & Class Names
# ─────────────────────────────────────────────

def validate_dataset_structure() -> bool:
    """Validate whichever dataset layout is active."""
    if use_flat_layout():
        classes = get_class_names()
        if not classes:
            logger.error("Flat dataset dir exists but contains no class sub-folders.")
            return False
        logger.info("Flat dataset validated: %d class(es).", len(classes))
        return True

    required = {"train": TRAIN_DIR, "valid": VALID_DIR, "test": TEST_DIR}
    valid = True
    for split, path in required.items():
        if not os.path.isdir(path):
            logger.error("Missing directory: '%s' — expected at '%s'", split, path)
            valid = False
            continue
        classes = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        if not classes:
            logger.error("No class sub-folders found in '%s'", path)
            valid = False
        else:
            logger.info("'%s' split -> %d class(es) found.", split, len(classes))
    return valid


def get_class_names() -> List[str]:
    """
    Read sorted class names from the active dataset root if present;
    otherwise gracefully fall back to config.CLASS_NAMES (e.g. in deployed web environments).
    """
    try:
        if use_flat_layout():
            root = FLAT_DATASET_DIR
        elif os.path.isdir(TRAIN_DIR):
            root = TRAIN_DIR
        else:
            root = None

        if root and os.path.isdir(root):
            classes = sorted(
                d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
            )
            if classes:
                logger.info("Detected %d class(es) from directory: %s", len(classes), classes)
                return classes
    except Exception as exc:
        logger.warning("Could not read class names from dataset directory: %s", exc)

    # Fallback for cloud deployment / web app inference
    from config import CLASS_NAMES
    logger.info("Using pre-defined class names (%d classes) for deployment/inference.", len(CLASS_NAMES))
    return CLASS_NAMES


# ─────────────────────────────────────────────
#  SHA256-Aware Stratified Splitting
# ─────────────────────────────────────────────

def _compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hex digest of a file in chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_split_csv(csv_path: Path, rows: List[dict]) -> None:
    """Write split manifest rows to a CSV file."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filepath", "class_index", "class_name", "sha256"]
        )
        writer.writeheader()
        writer.writerows(rows)


def _read_split_csv(csv_path: Path) -> List[dict]:
    """Read split manifest rows from a CSV file."""
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            r["class_index"] = int(r["class_index"])
            rows.append(r)
        return rows


def create_stratified_splits(
    dataset_dir: str = FLAT_DATASET_DIR,
    splits_dir: str = SPLITS_DIR,
    train_ratio: float = TRAIN_SPLIT,
    valid_ratio: float = VALID_SPLIT,
    test_ratio: float = TEST_SPLIT,
    seed: int = SEED,
    report_path: str = DATASET_AUDIT_REPORT,
) -> Tuple[List[dict], List[dict], List[dict], List[str], str]:
    """
    Scan all class directories, compute SHA256 hashes, perform a stratified
    80/10/10 split per class keeping duplicate hashes in the same split, save
    manifest CSVs to `data/splits/`, and generate a full audit report.

    Original images are NEVER moved, copied, or modified.
    """
    if abs((train_ratio + valid_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0.")

    root = Path(dataset_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    class_names = sorted(d.name for d in root.iterdir() if d.is_dir())
    if not class_names:
        raise ValueError(f"No class sub-directories found in {dataset_dir}")

    # 1. Scan all images into one common list with SHA256 hashes
    all_records: List[dict] = []
    hash_to_records: Dict[str, List[dict]] = defaultdict(list)

    for class_idx, class_name in enumerate(class_names):
        cls_dir = root / class_name
        img_files = sorted(
            f for f in cls_dir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
        )
        for img_path in img_files:
            digest = _compute_sha256(img_path)
            rec = {
                "filepath": str(img_path.resolve()),
                "class_index": class_idx,
                "class_name": class_name,
                "sha256": digest,
            }
            all_records.append(rec)
            hash_to_records[digest].append(rec)

    # 2. Stratified splitting per class, grouping by SHA256 so identical files
    #    within a class never leak across splits.
    rng = random.Random(seed)
    train_rows: List[dict] = []
    valid_rows: List[dict] = []
    test_rows: List[dict] = []

    # Track global hash -> assigned split to also prevent cross-class hash leakage
    global_hash_split: Dict[str, str] = {}

    for class_idx, class_name in enumerate(class_names):
        cls_records = [r for r in all_records if r["class_index"] == class_idx]

        # Group records by SHA256 within this class
        groups_by_hash: Dict[str, List[dict]] = defaultdict(list)
        for r in cls_records:
            groups_by_hash[r["sha256"]].append(r)

        # Separate groups already pinned by an earlier class's hash vs unassigned
        pinned_train, pinned_valid, pinned_test = [], [], []
        unassigned_groups = []
        for h, grp in groups_by_hash.items():
            if h in global_hash_split:
                target = global_hash_split[h]
                if target == "train":
                    pinned_train.extend(grp)
                elif target == "valid":
                    pinned_valid.extend(grp)
                else:
                    pinned_test.extend(grp)
            else:
                unassigned_groups.append((h, grp))

        rng.shuffle(unassigned_groups)

        total_cls = len(cls_records)
        target_val = max(1, round(total_cls * valid_ratio))
        target_test = max(1, round(total_cls * test_ratio))

        cls_train = list(pinned_train)
        cls_valid = list(pinned_valid)
        cls_test = list(pinned_test)

        for h, grp in unassigned_groups:
            if len(cls_valid) < target_val:
                cls_valid.extend(grp)
                global_hash_split[h] = "valid"
            elif len(cls_test) < target_test:
                cls_test.extend(grp)
                global_hash_split[h] = "test"
            else:
                cls_train.extend(grp)
                global_hash_split[h] = "train"

        train_rows.extend(cls_train)
        valid_rows.extend(cls_valid)
        test_rows.extend(cls_test)

    # Shuffle each split deterministically
    rng.shuffle(train_rows)
    rng.shuffle(valid_rows)
    rng.shuffle(test_rows)

    # 3. Save split manifests
    splits_path = Path(splits_dir)
    _write_split_csv(splits_path / "train.csv", train_rows)
    _write_split_csv(splits_path / "valid.csv", valid_rows)
    _write_split_csv(splits_path / "test.csv", test_rows)

    # 4. Run audit checks & build detailed report
    report = _build_audit_report(
        all_records=all_records,
        hash_to_records=hash_to_records,
        train_rows=train_rows,
        valid_rows=valid_rows,
        test_rows=test_rows,
        class_names=class_names,
        splits_path=splits_path,
    )

    if report_path:
        rp = Path(report_path)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(report, encoding="utf-8")
        logger.info("Dataset audit report saved to: %s", rp)

    return train_rows, valid_rows, test_rows, class_names, report


def _build_audit_report(
    all_records: List[dict],
    hash_to_records: Dict[str, List[dict]],
    train_rows: List[dict],
    valid_rows: List[dict],
    test_rows: List[dict],
    class_names: List[str],
    splits_path: Path,
) -> str:
    """Build a comprehensive dataset splitting and leakage audit report."""
    total_found = len(all_records)
    n_train, n_valid, n_test = len(train_rows), len(valid_rows), len(test_rows)

    train_paths = {r["filepath"] for r in train_rows}
    valid_paths = {r["filepath"] for r in valid_rows}
    test_paths = {r["filepath"] for r in test_rows}

    tv_path_overlap = len(train_paths & valid_paths)
    tt_path_overlap = len(train_paths & test_paths)
    vt_path_overlap = len(valid_paths & test_paths)

    train_hashes = {r["sha256"] for r in train_rows}
    valid_hashes = {r["sha256"] for r in valid_rows}
    test_hashes = {r["sha256"] for r in test_rows}

    tv_hash_overlap = len(train_hashes & valid_hashes)
    tt_hash_overlap = len(train_hashes & test_hashes)
    vt_hash_overlap = len(valid_hashes & test_hashes)

    dup_hashes = {h: recs for h, recs in hash_to_records.items() if len(recs) > 1}
    dup_extra_files = sum(len(recs) - 1 for recs in dup_hashes.values())

    train_counts = Counter(r["class_name"] for r in train_rows)
    valid_counts = Counter(r["class_name"] for r in valid_rows)
    test_counts = Counter(r["class_name"] for r in test_rows)

    missing_in_train = [c for c in class_names if train_counts[c] == 0]
    missing_in_valid = [c for c in class_names if valid_counts[c] == 0]
    missing_in_test = [c for c in class_names if test_counts[c] == 0]

    lines = [
        "=" * 78,
        "  PLANT DISEASE DETECTION — DATASET SPLIT & LEAKAGE AUDIT REPORT",
        "=" * 78,
        f"  Split manifests dir : {splits_path}",
        f"  Total images found  : {total_found:,}",
        f"  Training images     : {n_train:,} ({n_train / total_found * 100:.2f}%)",
        f"  Validation images   : {n_valid:,} ({n_valid / total_found * 100:.2f}%)",
        f"  Testing images      : {n_test:,} ({n_test / total_found * 100:.2f}%)",
        f"  Sum of splits       : {n_train + n_valid + n_test:,} / {total_found:,}",
        "",
        "── CLASS-WISE DISTRIBUTION ───────────────────────────────────────────────────",
        f"  {'#':<3} {'Class Name':<44} {'Train':>6} {'Valid':>6} {'Test':>6} {'Total':>6}",
        f"  {'-'*3} {'-'*44} {'-'*6} {'-'*6} {'-'*6} {'-'*6}",
    ]

    for idx, cls in enumerate(class_names, 1):
        tr = train_counts[cls]
        va = valid_counts[cls]
        te = test_counts[cls]
        lines.append(f"  {idx:<3} {cls:<44} {tr:>6} {va:>6} {te:>6} {tr + va + te:>6}")

    lines.extend([
        "",
        "── MISSING CLASS CHECK ───────────────────────────────────────────────────────",
        f"  Classes missing in train : {len(missing_in_train)} {missing_in_train}",
        f"  Classes missing in valid : {len(missing_in_valid)} {missing_in_valid}",
        f"  Classes missing in test  : {len(missing_in_test)} {missing_in_test}",
        "",
        "── DUPLICATE SHA256 HASH CHECK ───────────────────────────────────────────────",
        f"  Unique SHA256 hashes     : {len(hash_to_records):,}",
        f"  Duplicate hash groups    : {len(dup_hashes):,}",
        f"  Extra duplicate files    : {dup_extra_files:,}",
        f"  Cross-split hash overlap : train∩valid={tv_hash_overlap}, "
        f"train∩test={tt_hash_overlap}, valid∩test={vt_hash_overlap}",
        "",
        "── FILE PATH OVERLAP CHECK ───────────────────────────────────────────────────",
        f"  train ∩ valid paths      : {tv_path_overlap}",
        f"  train ∩ test paths       : {tt_path_overlap}",
        f"  valid ∩ test paths       : {vt_path_overlap}",
        "=" * 78,
    ])
    return "\n".join(lines)


# ─────────────────────────────────────────────
#  FLAT Layout — Streaming tf.data Pipelines
# ─────────────────────────────────────────────

def _decode_and_resize(filepath: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Read, decode, resize to IMAGE_SIZE, and normalise a single image to [0, 1]."""
    raw = tf.io.read_file(filepath)
    img = tf.io.decode_image(raw, channels=IMAGE_CHANNELS, expand_animations=False)
    img.set_shape([None, None, IMAGE_CHANNELS])
    img = tf.image.resize(img, IMAGE_SIZE, method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


def _make_tf_dataset(
    rows: List[dict],
    num_classes: int,
    batch_size: int,
    shuffle: bool = False,
    augment: bool = False,
) -> tf.data.Dataset:
    """Create a memory-safe streaming tf.data.Dataset (no .cache(), prefetch(1))."""
    filepaths = [r["filepath"] for r in rows]
    labels = tf.keras.utils.to_categorical(
        [r["class_index"] for r in rows], num_classes=num_classes
    )

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(rows), 1024), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(_decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)

    if augment:
        aug_layer = _build_augmentation_layer()
        ds = ds.map(
            lambda x, y: (aug_layer(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    # No .cache() — keeps RAM usage low on 8GB machines; prefetch(1) streams 1 batch ahead
    ds = ds.prefetch(1)
    return ds


def build_flat_datasets(
    dataset_dir: str = FLAT_DATASET_DIR,
    image_size: Tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    valid_split: float = VALID_SPLIT,
    test_split: float = TEST_SPLIT,
    augment_train: bool = True,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str]]:
    """
    Build leak-free, stratified train / validation / test tf.data.Datasets
    from `data/splits/{train,valid,test}.csv` (generating them if missing).
    """
    splits_path = Path(SPLITS_DIR)
    train_csv = splits_path / "train.csv"
    valid_csv = splits_path / "valid.csv"
    test_csv = splits_path / "test.csv"

    if train_csv.exists() and valid_csv.exists() and test_csv.exists():
        logger.info("Loading existing stratified split manifests from '%s'.", splits_path)
        train_rows = _read_split_csv(train_csv)
        valid_rows = _read_split_csv(valid_csv)
        test_rows = _read_split_csv(test_csv)
        class_names = get_class_names()
    else:
        logger.info("Split manifests not found — generating stratified 80/10/10 splits …")
        train_rows, valid_rows, test_rows, class_names, _ = create_stratified_splits(
            dataset_dir=dataset_dir,
            splits_dir=SPLITS_DIR,
            train_ratio=1.0 - valid_split - test_split,
            valid_ratio=valid_split,
            test_ratio=test_split,
            seed=SEED,
        )

    num_classes = len(class_names)
    logger.info(
        "Split sizes -> train: %d | valid: %d | test: %d | classes: %d",
        len(train_rows), len(valid_rows), len(test_rows), num_classes,
    )

    train_ds = _make_tf_dataset(
        train_rows, num_classes, batch_size, shuffle=True, augment=augment_train
    )
    val_ds = _make_tf_dataset(
        valid_rows, num_classes, batch_size, shuffle=False, augment=False
    )
    test_ds = _make_tf_dataset(
        test_rows, num_classes, batch_size, shuffle=False, augment=False
    )

    return train_ds, val_ds, test_ds, class_names


def _build_augmentation_layer() -> tf.keras.Sequential:
    """In-graph augmentation pipeline applied only to training batches."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(AUGMENTATION["rotation_range"] / 360),
        tf.keras.layers.RandomZoom(AUGMENTATION["zoom_range"]),
        tf.keras.layers.RandomTranslation(
            AUGMENTATION["height_shift_range"],
            AUGMENTATION["width_shift_range"],
        ),
        tf.keras.layers.RandomBrightness(
            factor=(
                AUGMENTATION["brightness_range"][0] - 1.0,
                AUGMENTATION["brightness_range"][1] - 1.0,
            ),
            value_range=(0.0, 1.0),
        ),
    ], name="augmentation")


# ─────────────────────────────────────────────
#  PRE-SPLIT Layout — ImageDataGenerator
# ─────────────────────────────────────────────

def build_train_generator(
    directory: str = TRAIN_DIR,
    batch_size: int = BATCH_SIZE,
):
    """Augmented ImageDataGenerator for pre-split train folder."""
    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=AUGMENTATION["rotation_range"],
        width_shift_range=AUGMENTATION["width_shift_range"],
        height_shift_range=AUGMENTATION["height_shift_range"],
        shear_range=AUGMENTATION["shear_range"],
        zoom_range=AUGMENTATION["zoom_range"],
        horizontal_flip=AUGMENTATION["horizontal_flip"],
        vertical_flip=AUGMENTATION["vertical_flip"],
        brightness_range=AUGMENTATION["brightness_range"],
        fill_mode=AUGMENTATION["fill_mode"],
    )
    gen = datagen.flow_from_directory(
        directory=directory,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        seed=SEED,
        shuffle=True,
    )
    logger.info("Train generator: %d images | %d classes", gen.samples, gen.num_classes)
    return gen


def build_val_generator(directory: str = VALID_DIR, batch_size: int = BATCH_SIZE):
    """Rescale-only validation generator for pre-split layout."""
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    gen = datagen.flow_from_directory(
        directory=directory,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        seed=SEED,
        shuffle=False,
    )
    logger.info("Val generator: %d images | %d classes", gen.samples, gen.num_classes)
    return gen


def build_test_generator(directory: str = TEST_DIR, batch_size: int = BATCH_SIZE):
    """Rescale-only test generator (no shuffle) for pre-split layout."""
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    gen = datagen.flow_from_directory(
        directory=directory,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        seed=SEED,
        shuffle=False,
    )
    logger.info("Test generator: %d images | %d classes", gen.samples, gen.num_classes)
    return gen


# ─────────────────────────────────────────────
#  Shared Utilities
# ─────────────────────────────────────────────

def compute_class_weights(class_names: List[str], dataset_dir: str = None) -> dict:
    """
    Compute balanced class weights from the training split only
    (never using validation or test data).
    """
    from sklearn.utils.class_weight import compute_class_weight

    train_csv = Path(SPLITS_DIR) / "train.csv"
    if use_flat_layout() and train_csv.exists():
        rows = _read_split_csv(train_csv)
        labels = np.array([r["class_index"] for r in rows])
    else:
        root = dataset_dir or (FLAT_DATASET_DIR if use_flat_layout() else TRAIN_DIR)
        labels_list = []
        for idx, cls in enumerate(class_names):
            cls_path = Path(root) / cls
            if cls_path.is_dir():
                count = sum(
                    1 for f in cls_path.iterdir() if f.suffix.lower() in SUPPORTED_EXTS
                )
                labels_list.extend([idx] * count)
        labels = np.array(labels_list)

    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    class_weight_dict = dict(zip(classes.tolist(), weights.tolist()))
    logger.info("Class weights computed for %d classes (from train split).", len(class_weight_dict))
    return class_weight_dict


def dataset_summary(ds: tf.data.Dataset, name: str = "Dataset") -> None:
    """Log a human-readable summary of a tf.data.Dataset."""
    batches = ds.cardinality().numpy()
    logger.info("── %s Summary ──", name)
    logger.info("  Batches       : %d", batches)
    logger.info("  Batch size    : %d", BATCH_SIZE)
    logger.info("  ~Images       : %d", batches * BATCH_SIZE)
    logger.info("  Image size    : %s", IMAGE_SIZE)
