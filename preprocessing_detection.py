import os
import json
import shutil
import numpy as np
from PIL import Image, ImageDraw

# -----------------------------------------------------------------------
# IMPORTANT: Run preprocessing.py (classification) FIRST.
# This script reads the exact image splits and label_map that it produced,
# guaranteeing both tasks use the identical train/val/test images.
# -----------------------------------------------------------------------

CLASSIFICATION_OUTPUT_DIR = "processed_dataset"
OUTPUT_DIR = "processed_dataset"

TRAIN_ANN_DIR = "data/train/annos"
VAL_ANN_DIR = "data/validation/annos"
TEST_ANN_DIR = "data/test/json_for_test"

ALL_ANN_DIRS = [TRAIN_ANN_DIR, VAL_ANN_DIR, TEST_ANN_DIR]

ALL_IMAGE_DIRS = [
    "data/train/image",
    "data/validation/image",
    "data/test/test",
]

CATEGORY_NAMES = {
    1:  "short_sleeve_top",
    2:  "long_sleeve_top",
    3:  "short_sleeve_outwear",
    4:  "long_sleeve_outwear",
    5:  "vest",
    6:  "sling",
    7:  "shorts",
    8:  "trousers",
    9:  "skirt",
    10: "short_sleeve_dress",
    11: "long_sleeve_dress",
    12: "vest_dress",
    13: "sling_dress",
}

# ================================================
# STEP 1 — Load label_map and splits
# ================================================

label_map_path = os.path.join(CLASSIFICATION_OUTPUT_DIR, "label_map.json")

if not os.path.exists(label_map_path):
    raise FileNotFoundError(
        f"label_map.json not found at {label_map_path}.\n"
        "Please run preprocessing.py (classification) first."
    )

with open(label_map_path) as f:
    meta = json.load(f)

top5 = meta["top5_original_ids"]
label_map = {int(k): int(v) for k, v in meta["label_map"].items()}
NUM_CLASSES = meta["num_classes"]

print("Loaded label_map from classification output.")
print(f"  Top-5 categories : {top5}")
print(f"  Label map        : {label_map}")
print(f"  Num classes      : {NUM_CLASSES}")

# Load splits — image paths may be absolute, we normalise to absolute using
# the stored path directly (it already points to the real file).
split_map = {}

for split_name in ["train", "val", "test"]:
    split_path = os.path.join(CLASSIFICATION_OUTPUT_DIR, f"{split_name}.json")

    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"{split_name}.json not found. Run preprocessing.py first."
        )

    with open(split_path) as f:
        records = json.load(f)

    split_map[split_name] = [r["image"] for r in records]
    print(f"  Loaded {split_name} split : {len(split_map[split_name])} images")

# ================================================
# STEP 2 — Build lookup: filename (stem) → ann_path
#                        filename (stem) → real image_path
#
# We key on just the bare filename e.g. "137724" so it matches
# regardless of whether the stored path is absolute or relative.
# ================================================

print("\nBuilding filename → annotation/image lookups...")

# stem → annotation json path
stem_to_ann = {}

for ann_dir in ALL_ANN_DIRS:
    if not os.path.isdir(ann_dir):
        print(f"  Skipping missing annotation dir: {ann_dir}")
        continue
    for ann_file in os.listdir(ann_dir):
        if not ann_file.endswith(".json"):
            continue
        stem = os.path.splitext(ann_file)[0]          # "137724"
        stem_to_ann[stem] = os.path.join(ann_dir, ann_file)

print(f"  Annotation stems indexed : {len(stem_to_ann)}")

# stem → real (existing) image path  — needed for YOLO image copy & size reading
stem_to_img = {}

for img_dir in ALL_IMAGE_DIRS:
    if not os.path.isdir(img_dir):
        continue
    for img_file in os.listdir(img_dir):
        stem = os.path.splitext(img_file)[0]
        stem_to_img[stem] = os.path.join(img_dir, img_file)

print(f"  Image stems indexed      : {len(stem_to_img)}")

# ================================================
# HELPER FUNCTIONS
# ================================================


def stem_from_path(path):
    """Extract bare filename without extension from any path (abs or rel)."""
    return os.path.splitext(os.path.basename(path))[0]


def get_image_size(img_path):
    try:
        with Image.open(img_path) as im:
            return im.width, im.height
    except Exception:
        return 0, 0


def polygon_area(poly):
    n = len(poly) // 2
    if n < 3:
        return 0.0
    xs = [poly[2 * i] for i in range(n)]
    ys = [poly[2 * i + 1] for i in range(n)]
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j]
        area -= xs[j] * ys[i]
    return abs(area) / 2.0


def parse_annotations(ann_path):
    """
    Parse one annotation JSON.
    Returns list of instance dicts — only top-5 categories kept.
    Each dict: {category_id (0-4), original_category_id, bbox [x1,y1,x2,y2], segmentation}
    """
    with open(ann_path) as f:
        data = json.load(f)

    instances = []

    for key in data:
        if not key.startswith("item"):
            continue
        item = data[key]
        orig_cat = item.get("category_id")
        if orig_cat not in label_map:
            continue
        bbox = item.get("bounding_box")
        segs = item.get("segmentation") or []
        if bbox is None:
            continue
        instances.append({
            "category_id":          label_map[orig_cat],
            "original_category_id": orig_cat,
            "bbox":                 bbox,
            "segmentation":         segs,
        })

    return instances


def resolve(stored_path):
    """
    Given a stored image path (may be absolute or relative),
    return (stem, ann_path, real_img_path) or None if not found.
    """
    stem = stem_from_path(stored_path)
    ann_path = stem_to_ann.get(stem)
    img_path = stem_to_img.get(stem)

    # Prefer the stored path if it actually exists (it's absolute and valid)
    if os.path.exists(stored_path):
        img_path = stored_path

    if ann_path is None or img_path is None:
        return None

    return stem, ann_path, img_path


# ================================================
# OUTPUT A — Generic Detection JSON
# ================================================

def save_detection_json(split_name, image_list):

    out_path = os.path.join(OUTPUT_DIR, f"{split_name}_detection.json")
    records = []
    skipped = 0

    for stored_path in image_list:
        resolved = resolve(stored_path)
        if resolved is None:
            skipped += 1
            continue
        _, ann_path, img_path = resolved
        instances = parse_annotations(ann_path)
        if not instances:
            skipped += 1
            continue
        records.append({"image": img_path, "annotations": instances})

    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"  [{split_name}] Generic JSON -> {out_path}  "
          f"({len(records)} images, {skipped} skipped)")


# ================================================
# OUTPUT B — COCO-style JSON  (Mask R-CNN)
# ================================================

def save_coco_json(split_name, image_list):

    out_path = os.path.join(OUTPUT_DIR, f"{split_name}_coco.json")

    categories = sorted([
        {
            "id":   label_map[orig] + 1,
            "name": CATEGORY_NAMES.get(orig, str(orig)),
        }
        for orig in top5
    ], key=lambda x: x["id"])

    images_out = []
    anns_out = []
    image_id = 1
    ann_id = 1
    skipped = 0

    for stored_path in image_list:
        resolved = resolve(stored_path)
        if resolved is None:
            skipped += 1
            continue
        _, ann_path, img_path = resolved
        instances = parse_annotations(ann_path)
        if not instances:
            skipped += 1
            continue

        w, h = get_image_size(img_path)

        images_out.append({
            "id":        image_id,
            "file_name": img_path,
            "width":     w,
            "height":    h,
        })

        for inst in instances:
            x1, y1, x2, y2 = inst["bbox"]
            bw = x2 - x1
            bh = y2 - y1
            area = sum(polygon_area(p)
                       for p in inst["segmentation"]) or float(bw * bh)

            anns_out.append({
                "id":           ann_id,
                "image_id":     image_id,
                "category_id":  inst["category_id"] + 1,
                "bbox":         [x1, y1, bw, bh],
                "area":         area,
                "segmentation": inst["segmentation"],
                "iscrowd":      0,
            })
            ann_id += 1

        image_id += 1

    with open(out_path, "w") as f:
        json.dump({"images": images_out, "annotations": anns_out,
                  "categories": categories}, f, indent=2)

    print(f"  [{split_name}] COCO JSON -> {out_path}  "
          f"({len(images_out)} images, {len(anns_out)} annotations, {skipped} skipped)")


# ================================================
# OUTPUT C — YOLO format  (YOLOv8)
# ================================================

def save_yolo(split_map_local):

    yolo_root = os.path.join(OUTPUT_DIR, "yolo_dataset")

    for split_name, image_list in split_map_local.items():

        img_out = os.path.join(yolo_root, "images", split_name)
        lbl_out = os.path.join(yolo_root, "labels", split_name)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        count = 0
        skipped = 0

        for stored_path in image_list:
            resolved = resolve(stored_path)
            if resolved is None:
                skipped += 1
                continue
            stem, ann_path, img_path = resolved
            instances = parse_annotations(ann_path)
            if not instances:
                skipped += 1
                continue

            w, h = get_image_size(img_path)
            if w == 0 or h == 0:
                skipped += 1
                continue

            # Copy image
            img_fname = os.path.basename(img_path)
            dst_img = os.path.join(img_out, img_fname)
            if not os.path.exists(dst_img):
                shutil.copy2(img_path, dst_img)

            # Write label file
            lbl_path = os.path.join(lbl_out, stem + ".txt")
            lines = []

            for inst in instances:
                cls = inst["category_id"]
                x1, y1, x2, y2 = inst["bbox"]

                xc = ((x1 + x2) / 2) / w
                yc = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

                for poly in inst["segmentation"]:
                    if len(poly) < 6:
                        continue
                    pts = []
                    for k in range(0, len(poly), 2):
                        pts.append(f"{poly[k] / w:.6f}")
                        pts.append(f"{poly[k+1] / h:.6f}")
                    lines.append(f"{cls} " + " ".join(pts))

            with open(lbl_path, "w") as f:
                f.write("\n".join(lines))

            count += 1

        print(
            f"  [YOLO {split_name}] {count} images written, {skipped} skipped -> {img_out}")

    class_names = [CATEGORY_NAMES.get(c, str(c)) for c in top5]
    yaml_content = "\n".join([
        f"path: {os.path.abspath(yolo_root)}",
        "train: images/train",
        "val:   images/val",
        "test:  images/test",
        "",
        f"nc: {NUM_CLASSES}",
        f"names: {class_names}",
    ])
    with open(os.path.join(yolo_root, "dataset.yaml"), "w") as f:
        f.write(yaml_content)
    print(f"  [YOLO] dataset.yaml -> {yolo_root}/dataset.yaml")


# ================================================
# OUTPUT D — U-Net semantic segmentation masks
# ================================================

def save_unet_masks(split_map_local):

    mask_root = os.path.join(OUTPUT_DIR, "unet_masks")

    for split_name, image_list in split_map_local.items():

        mask_out = os.path.join(mask_root, split_name)
        os.makedirs(mask_out, exist_ok=True)

        count = 0
        skipped = 0

        for stored_path in image_list:
            resolved = resolve(stored_path)
            if resolved is None:
                skipped += 1
                continue
            stem, ann_path, img_path = resolved
            instances = parse_annotations(ann_path)
            if not instances:
                skipped += 1
                continue

            w, h = get_image_size(img_path)
            if w == 0 or h == 0:
                skipped += 1
                continue

            pil_mask = Image.fromarray(np.zeros((h, w), dtype=np.uint8))
            draw = ImageDraw.Draw(pil_mask)

            for inst in instances:
                pixel_val = inst["category_id"] + 1
                for poly in inst["segmentation"]:
                    if len(poly) < 6:
                        continue
                    coords = [(poly[k], poly[k+1])
                              for k in range(0, len(poly), 2)]
                    draw.polygon(coords, fill=pixel_val)

            pil_mask.save(os.path.join(mask_out, stem + ".png"))
            count += 1

        print(
            f"  [U-Net {split_name}] {count} masks saved, {skipped} skipped -> {mask_out}")


# ================================================
# STEP 3 — Run all outputs
# ================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n--- Generating Generic Detection JSONs ---")
for split_name, image_list in split_map.items():
    save_detection_json(split_name, image_list)

print("\n--- Generating COCO JSONs (Mask R-CNN) ---")
for split_name, image_list in split_map.items():
    save_coco_json(split_name, image_list)

print("\n--- Generating YOLO Dataset ---")
save_yolo(split_map)

print("\n--- Generating U-Net Semantic Masks ---")
save_unet_masks(split_map)

print("\nDetection preprocessing completed successfully.")
print(f"\nOutputs in: {OUTPUT_DIR}/")
print("  train/val/test _detection.json  — generic per-instance format")
print("  train/val/test _coco.json       — COCO format for Mask R-CNN")
print("  yolo_dataset/                   — images + labels + dataset.yaml")
print("  unet_masks/                     — grayscale PNG masks (pixel = class index)")
