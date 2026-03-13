import os
import json
import random
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split

TRAIN_IMAGE_DIR = "data/train/image"
TRAIN_ANN_DIR = "data/train/annos"

VAL_IMAGE_DIR = "data/validation/image"
VAL_ANN_DIR = "data/validation/annos"

TEST_IMAGE_DIR = "data/test/test"
TEST_ANN_DIR = "data/test/json_for_test"

DATA_SPLITS = [
    (TRAIN_IMAGE_DIR, TRAIN_ANN_DIR),
    (VAL_IMAGE_DIR, VAL_ANN_DIR),
    (TEST_IMAGE_DIR, TEST_ANN_DIR)
]

OUTPUT_DIR = "processed_dataset"
TARGET_RATIO = 0.6

random.seed(42)

# Count category frequencies
category_counter = Counter()
image_to_categories = {}

print("Scanning dataset annotations...")

for IMAGE_DIR, ANN_DIR in DATA_SPLITS:

    for ann_file in os.listdir(ANN_DIR):

        if not ann_file.endswith(".json"):
            continue

        ann_path = os.path.join(ANN_DIR, ann_file)

        with open(ann_path) as f:
            data = json.load(f)

        cats = set()

        for key in data:

            if key.startswith("item"):

                cat = data[key]["category_id"]
                cats.add(cat)

        if len(cats) == 0:
            continue

        image_name = ann_file.replace(".json", ".jpg")
        image_path = os.path.abspath(os.path.join(IMAGE_DIR, image_name))

        if not os.path.exists(image_path):
            continue

        image_to_categories[image_path] = list(cats)

        for c in cats:
            category_counter[c] += 1


print("\nCategory frequencies:")
print(category_counter)

# Select top 5 categories
top5 = [c for c, _ in category_counter.most_common(5)]

print("\nTop 5 categories:", top5)

label_map = {cat: i for i, cat in enumerate(top5)}
NUM_CLASSES = len(top5)

# Filter images containing top-5 classes
filtered_images = []

for img, cats in image_to_categories.items():

    filtered = [c for c in cats if c in top5]

    if len(filtered) > 0:
        filtered_images.append((img, filtered))


print("\nImages after filtering:", len(filtered_images))

# Create multi-label vectors
dataset = []

for img, cats in filtered_images:

    label_vector = [0] * NUM_CLASSES

    for c in cats:
        label_vector[label_map[c]] = 1

    dataset.append({
        "image": img,
        "labels": label_vector
    })


print("Dataset entries created:", len(dataset))

# Balanced subset sampling
target_size = int(len(dataset) * TARGET_RATIO)

print("\nCreating balanced subset:", target_size)

class_buckets = defaultdict(list)

for sample in dataset:

    for idx, val in enumerate(sample["labels"]):

        if val == 1:
            class_buckets[idx].append(sample)


subset = []
seen = set()

for cls in class_buckets:

    random.shuffle(class_buckets[cls])

    target_per_class = target_size // NUM_CLASSES
    count = 0

    for sample in class_buckets[cls]:

        if sample["image"] not in seen:

            subset.append(sample)
            seen.add(sample["image"])
            count += 1

        if count >= target_per_class:
            break


print("Subset size:", len(subset))

# Train / Validation / Test split
images = [s["image"] for s in subset]
labels = [s["labels"] for s in subset]

train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
    images,
    labels,
    test_size=0.30,
    random_state=42
)

val_imgs, test_imgs, val_labels, test_labels = train_test_split(
    temp_imgs,
    temp_labels,
    test_size=0.5,
    random_state=42
)

print("\nSplit sizes:")
print("Train:", len(train_imgs))
print("Validation:", len(val_imgs))
print("Test:", len(test_imgs))

# Save splits
os.makedirs(OUTPUT_DIR, exist_ok=True)

splits = {
    "train": (train_imgs, train_labels),
    "val": (val_imgs, val_labels),
    "test": (test_imgs, test_labels)
}

for split in splits:

    imgs, lbls = splits[split]

    data = []

    for img, lbl in zip(imgs, lbls):

        data.append({
            "image": img,
            "labels": lbl
        })

    out_path = os.path.join(OUTPUT_DIR, f"{split}.json")

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"{split} file saved → {out_path}")


print("\nPreprocessing completed successfully.")

# Save label_map so preprocessing_detection.py can read the exact same top-5
label_map_out = {
    "top5_original_ids": top5,
    "label_map":         {str(k): v for k, v in label_map.items()},
    "category_names":    {str(c): str(c) for c in top5},
    "num_classes":       NUM_CLASSES,
}
with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
    json.dump(label_map_out, f, indent=2)

print("label_map.json saved.")
