"""
inspect_dataset.py  — Read-only inspection script.
Produces a full report on dataset structure, image counts,
formats, and project compatibility.
"""

import os, sys, json
from pathlib import Path
from collections import defaultdict

# ── Locate dataset ─────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
CANDIDATES  = [
    BASE_DIR / "PlantVillage" / "PlantVillage",
    BASE_DIR / "PlantVillage",
    BASE_DIR / "dataset" / "train",
    BASE_DIR.parent / "PlantVillage" / "PlantVillage",
    BASE_DIR.parent / "PlantVillage",
]

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

def find_dataset_root(candidates):
    for c in candidates:
        c = Path(c)
        if c.is_dir():
            subdirs = [d for d in c.iterdir() if d.is_dir()]
            if subdirs:
                # Quick check: does at least one sub-dir contain images?
                for sd in subdirs[:3]:
                    imgs = [f for f in sd.iterdir()
                            if f.suffix.lower() in SUPPORTED_EXTS]
                    if imgs:
                        return c
    return None

dataset_root = find_dataset_root(CANDIDATES)

print("=" * 65)
print("  PLANT DISEASE DETECTION — DATASET INSPECTION REPORT")
print("=" * 65)

if dataset_root is None:
    print("\n❌  Could not locate dataset. Searched paths:")
    for c in CANDIDATES:
        print(f"    {c}  → {'EXISTS (no images)' if Path(c).exists() else 'NOT FOUND'}")
    sys.exit(1)

print(f"\n✅  Dataset root found : {dataset_root}\n")

# ── Scan class folders ─────────────────────────────────────────────────────────
class_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir()])
print(f"── 1. CLASS FOLDERS ({'─'*45})")
print(f"   Total classes found : {len(class_dirs)}\n")

total_images   = 0
ext_counter    = defaultdict(int)
corrupt_images = []
class_stats    = []

for cls_dir in class_dirs:
    images = [f for f in cls_dir.iterdir() if f.suffix.lower() in SUPPORTED_EXTS]
    for img_path in images:
        ext_counter[img_path.suffix.lower()] += 1
        # Corruption check via PIL
        try:
            from PIL import Image
            with Image.open(img_path) as im:
                im.verify()
        except Exception as e:
            corrupt_images.append((str(img_path), str(e)))
    class_stats.append((cls_dir.name, len(images)))
    total_images += len(images)

# Print class table
print(f"   {'#':<4} {'Class Name':<50} {'Images':>7}")
print(f"   {'─'*4} {'─'*50} {'─'*7}")
for i, (name, count) in enumerate(class_stats, 1):
    print(f"   {i:<4} {name:<50} {count:>7}")

print(f"\n   {'TOTAL':<55} {total_images:>7}")

# ── Image format summary ───────────────────────────────────────────────────────
print(f"\n── 2. IMAGE FORMATS ({'─'*45})")
for ext, count in sorted(ext_counter.items(), key=lambda x: -x[1]):
    print(f"   {ext:<10} : {count:>6} images")

# ── Corruption check ───────────────────────────────────────────────────────────
print(f"\n── 3. CORRUPTION CHECK ({'─'*42})")
if corrupt_images:
    print(f"   ⚠️  {len(corrupt_images)} corrupt image(s) found:")
    for path, err in corrupt_images[:10]:
        print(f"   ✗  {Path(path).name}  →  {err}")
    if len(corrupt_images) > 10:
        print(f"   ... and {len(corrupt_images)-10} more.")
else:
    print(f"   ✅  No corrupt images found.")

# ── Sample image dimensions ────────────────────────────────────────────────────
print(f"\n── 4. SAMPLE IMAGE DIMENSIONS ({'─'*35})")
sizes = []
for cls_dir in class_dirs[:5]:
    imgs = [f for f in cls_dir.iterdir() if f.suffix.lower() in SUPPORTED_EXTS]
    for img_path in imgs[:3]:
        try:
            from PIL import Image
            with Image.open(img_path) as im:
                sizes.append((im.size, im.mode, cls_dir.name))
        except:
            pass
unique_sizes = set(s[0] for s in sizes)
unique_modes = set(s[1] for s in sizes)
print(f"   Sampled {len(sizes)} images across {min(5,len(class_dirs))} classes")
print(f"   Unique sizes found : {unique_sizes}")
print(f"   Color modes found  : {unique_modes}")

# ── Project expectation vs dataset structure ───────────────────────────────────
print(f"\n── 5. PROJECT vs DATASET STRUCTURE ({'─'*29})")
project_train = BASE_DIR / "dataset" / "train"
project_valid = BASE_DIR / "dataset" / "valid"
project_test  = BASE_DIR / "dataset" / "test"

print(f"   Project expects:")
print(f"     dataset/train/  → {'✅ EXISTS' if project_train.exists() else '❌ EMPTY / MISSING'}")
print(f"     dataset/valid/  → {'✅ EXISTS' if project_valid.exists() else '❌ EMPTY / MISSING'}")
print(f"     dataset/test/   → {'✅ EXISTS' if project_test.exists() else '❌ EMPTY / MISSING'}")

train_classes = [d.name for d in project_train.iterdir() if d.is_dir()] if project_train.exists() else []
print(f"\n   dataset/train/ class folders: {len(train_classes)}")

print(f"\n   Dataset currently at   : {dataset_root}")
print(f"   Structure type         : FLAT (all classes in one folder, NO train/valid/test split)")

# ── Recommended next steps ────────────────────────────────────────────────────
print(f"\n── 6. WHAT NEEDS TO HAPPEN ({'─'*38})")
print("""
   The project's config.py points to:
     → dataset/train/   (for training)
     → dataset/valid/   (for validation)
     → dataset/test/    (for testing)

   But the extracted PlantVillage dataset has a FLAT structure:
     → PlantVillage/<ClassName>/*.jpg  (all images in one place)

   SOLUTION (no images moved, no originals touched):
   ─────────────────────────────────────────────────
   Create symlinks OR use Keras's validation_split to split on-the-fly.

   RECOMMENDED APPROACH  ★
   Use tf.keras.utils.image_dataset_from_directory() with
   validation_split=0.2 directly on the flat PlantVillage folder.
   This avoids touching or duplicating any original files.

   ALTERNATIVE (if symlinks preferred):
   Update TRAIN_DIR in config.py to point directly to the
   PlantVillage folder, and let data_loader.py handle the split.
""")

print("=" * 65)
print(f"  SUMMARY")
print("=" * 65)
print(f"  Dataset root   : {dataset_root}")
print(f"  Total classes  : {len(class_dirs)}")
print(f"  Total images   : {total_images:,}")
print(f"  Formats        : {', '.join(ext_counter.keys())}")
print(f"  Corrupt files  : {len(corrupt_images)}")
print(f"  Split needed   : YES  (flat → train/valid/test)")
print(f"  Files to touch : config.py + data_loader.py only")
print("=" * 65)

