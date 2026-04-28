import numpy as np
import pandas as pd
import pickle
import re
import os
import json
from PIL import Image

# ── Config ─────────────────────────────────────────────────────────────
PICKLE_PATH = "/Users/izzy/Documents/dataanalysis/Segmentation_Evaluation/segmentation_results.pickle"
NPY_DIR     = "/Users/izzy/Documents/dataanalysis/Segmentation_Evaluation/brush_strokes/Izzy"
JSON_PATH   = "/Users/izzy/Documents/dataanalysis/Segmentation_Evaluation/tasks.json"

# ── Build task_id → image_name mapping ────────────────────────────────
with open(JSON_PATH, "r") as f:
    tasks = json.load(f)

# Task IDs corresponding to each image in order (non-sequential due to deletions)
#TASK_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
TASK_IDS = [1, 2, 3, 4, 6, 7, 10, 11, 14, 15, 21, 24] #Izzy's original list with deletions

task_to_image = {}
for task_id, task in zip(TASK_IDS, tasks):
    raw_path = task["image"]
    image_name = raw_path.split("-", 1)[1]  # strips the hash prefix e.g. "058e7ab7-"
    task_to_image[task_id] = image_name

# ── Load predictions ───────────────────────────────────────────────────
with open(PICKLE_PATH, "rb") as f:
    df = pickle.load(f)
predictions = {row["img_name"]: np.array(row["predicted_segmentation"])
               for _, row in df.iterrows()}

# ── Parse npy filenames ────────────────────────────────────────────────
def parse_npy_filename(filename):
    # Extract task number and class id from e.g.:
    # task-1-annotation-1-by-1-tag-"1": "building",-0.npy
    task_m = re.search(r'task-(\d+)', filename)
    class_m = re.search(r'"(\d+)"', filename)
    if not class_m:
        class_m = re.search(r'tag-(\d+)_', filename)
        class_id = int(class_m.group(1)) if class_m else None
    else:
        class_id = int(class_m.group(1))
    task_id = int(task_m.group(1)) if task_m else None
    return task_id, class_id

# ── IoU ────────────────────────────────────────────────────────────────
def compute_iou(gt_mask, pred_mask, class_id):
    H, W = pred_mask.shape
    gt_resized  = np.array(Image.fromarray(gt_mask).resize((W, H), Image.NEAREST))
    gt_binary   = (gt_resized > 0)
    pred_binary = (pred_mask == class_id)
    intersection = int(np.logical_and(gt_binary, pred_binary).sum())
    union        = int(np.logical_or(gt_binary, pred_binary).sum())
    iou = intersection / union if union > 0 else float("nan")
    return iou, intersection, union

# ── Run ────────────────────────────────────────────────────────────────
results = []

for fname in os.listdir(NPY_DIR):
    if not fname.endswith(".npy"):
        continue

    task_id, class_id = parse_npy_filename(fname)

    if task_id is None:
        print(f"Could not parse task from: {fname}")
        continue
    if class_id is None:
        print(f"Could not parse class from: {fname}")
        continue

    image_name = task_to_image.get(task_id)
    if image_name is None:
        print(f"No image mapping for task {task_id}: {fname}")
        continue

    if image_name not in predictions:
        print(f"No prediction found for image {image_name} (task {task_id})")
        continue

    pred_mask = predictions[image_name]
    gt_mask   = np.load(os.path.join(NPY_DIR, fname))
    iou, inter, uni = compute_iou(gt_mask, pred_mask, class_id)

    print(f"{image_name} | task {task_id} | class {class_id:3d} | IoU: {iou:.4f}")
    results.append({"image": image_name, "task": task_id, "class": class_id,
                    "iou": iou, "intersection": inter, "union": uni})

results_df = pd.DataFrame(results).sort_values(["image", "class"])

def micro_iou(group):
    return group["intersection"].sum() / group["union"].sum() if group["union"].sum() > 0 else float("nan")

print("\n── Per-class IoU ──")
per_class_micro = {}
for cls, grp in results_df.groupby("class"):
    per_class_micro[cls] = micro_iou(grp)
    print(f"  class {cls:3d}  |  mean {grp['iou'].mean():.4f}  |  median {grp['iou'].median():.4f}  |  micro {per_class_micro[cls]:.4f}  (n={len(grp)})")

miou = np.nanmean(list(per_class_micro.values()))
print(f"\n── mIoU (classic, mean of per-class IoU) ──")
print(f"  mIoU = {miou:.4f}  (averaged over {len(per_class_micro)} classes)")


print("\n── Per-image IoU ──")
for img, grp in results_df.groupby("image"):
    print(f"  {img}  |  mean {grp['iou'].mean():.4f}  |  median {grp['iou'].median():.4f}  |  micro {micro_iou(grp):.4f}")

print(f"\n── Overall ──")
print(f"  mean   {results_df['iou'].mean():.4f}")
print(f"  median {results_df['iou'].median():.4f}")
print(f"  micro  {micro_iou(results_df):.4f}")

# ── Combined "greenery" class (plant=17, grass=9, tree=4) ─────────────
GREENERY_IDS = {4, 9, 17}
BUILT_IDS = {1, 6, 11, 84}

# Group GT masks by task_id for the greenery classes
greenery_gt_by_task = {}
for fname in os.listdir(NPY_DIR):
    if not fname.endswith(".npy"):
        continue
    task_id, class_id = parse_npy_filename(fname)
    if task_id is None or class_id is None:
        continue
    if class_id in GREENERY_IDS:
        mask = np.load(os.path.join(NPY_DIR, fname))
        if task_id not in greenery_gt_by_task:
            greenery_gt_by_task[task_id] = mask > 0
        else:
            greenery_gt_by_task[task_id] = greenery_gt_by_task[task_id] | (mask > 0)

greenery_results = []
for task_id, gt_combined in greenery_gt_by_task.items():
    image_name = task_to_image.get(task_id)
    if image_name is None or image_name not in predictions:
        continue
    pred_mask = predictions[image_name]
    H, W = pred_mask.shape
    gt_resized = np.array(Image.fromarray(gt_combined).resize((W, H), Image.NEAREST))
    pred_greenery = np.isin(pred_mask, list(GREENERY_IDS))
    intersection = int(np.logical_and(gt_resized, pred_greenery).sum())
    union = int(np.logical_or(gt_resized, pred_greenery).sum())
    iou = intersection / union if union > 0 else float("nan")
    print(f"{image_name} | task {task_id} | greenery (4+9+17) | IoU: {iou:.4f}")
    greenery_results.append({"image": image_name, "task": task_id,
                             "iou": iou, "intersection": intersection, "union": union})

greenery_df = pd.DataFrame(greenery_results)
if not greenery_df.empty:
    g_micro = greenery_df["intersection"].sum() / greenery_df["union"].sum() if greenery_df["union"].sum() > 0 else float("nan")
    print(f"\n── Combined greenery (tree+grass+plant) ──")
    print(f"  mean   {greenery_df['iou'].mean():.4f}")
    print(f"  median {greenery_df['iou'].median():.4f}")
    print(f"  micro  {g_micro:.4f}")


# Group GT masks by task_id for the built classes
built_gt_by_task = {}
for fname in os.listdir(NPY_DIR):
    if not fname.endswith(".npy"):
        continue
    task_id, class_id = parse_npy_filename(fname)
    if task_id is None or class_id is None:
        continue
    if class_id in BUILT_IDS:
        mask = np.load(os.path.join(NPY_DIR, fname))
        if task_id not in built_gt_by_task:
            built_gt_by_task[task_id] = mask > 0
        else:
             built_gt_by_task[task_id] = built_gt_by_task[task_id] | (mask > 0)

built_results = []
for task_id, gt_combined in built_gt_by_task.items():
    image_name = task_to_image.get(task_id)
    if image_name is None or image_name not in predictions:
        continue
    pred_mask = predictions[image_name]
    H, W = pred_mask.shape
    gt_resized = np.array(Image.fromarray(gt_combined).resize((W, H), Image.NEAREST))
    pred_built = np.isin(pred_mask, list(BUILT_IDS))
    intersection = int(np.logical_and(gt_resized, pred_built).sum())
    union = int(np.logical_or(gt_resized, pred_built).sum())
    iou = intersection / union if union > 0 else float("nan")
    print(f"{image_name} | task {task_id} | built (1+6+11+84) | IoU: {iou:.4f}")
    built_results.append({"image": image_name, "task": task_id,
                          "iou": iou, "intersection": intersection, "union": union})

built_df = pd.DataFrame(built_results)
if not built_df.empty:
    b_micro = built_df["intersection"].sum() / built_df["union"].sum() if built_df["union"].sum() > 0 else float("nan")
    print(f"\n── Combined built (building+sky+sidewalk+person+car) ──")
    print(f"  mean   {built_df['iou'].mean():.4f}")
    print(f"  median {built_df['iou'].median():.4f}")
    print(f"  micro  {b_micro:.4f}")
# ── mIoU restricted to classes used in the paper ──────────────────────
# 5 classes: building, sky, person, car (separate) + greenery (tree+grass+plant combined)
PAPER_CLASSES_SEPARATE = {2: "sky", 12: "person", 20: "car"}
paper_micro = {cls: per_class_micro[cls] for cls in PAPER_CLASSES_SEPARATE if cls in per_class_micro}
missing = [f"{PAPER_CLASSES_SEPARATE[c]} ({c})" for c in PAPER_CLASSES_SEPARATE if c not in per_class_micro]
if missing:
    print(f"  (no annotations found for: {', '.join(missing)})")

print(f"\n── mIoU (paper classes) ──")
for cls, iou_val in paper_micro.items():
    print(f"  {PAPER_CLASSES_SEPARATE[cls]:10s} (class {cls:2d})  |  IoU: {iou_val:.4f}")
if not built_df.empty:
    paper_micro["built"] = b_micro
    print(f"  {'built':10s} (1+6+11+84) |  IoU: {b_micro:.4f}")
if not greenery_df.empty:
    paper_micro["greenery"] = g_micro
    print(f"  {'greenery':10s} (4+9+17)    |  IoU: {g_micro:.4f}")
miou_paper = np.nanmean(list(paper_micro.values()))
print(f"  mIoU = {miou_paper:.4f}  (averaged over {len(paper_micro)} classes)")
