import numpy as np
import pandas as pd
import pickle
import re
import os
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Config ─────────────────────────────────────────────────────────────
PICKLE_PATH  = "~/Segmentation_Evaluation/segmentation_results.pickle"
NPY_DIR      = "~/Segmentation_Evaluation/brush_strokes/Rater1/" #adjust here to Rater2
JSON_PATH    = "~/Segmentation_Evaluation/config/tasks_rater1.json" #adjust here to Rater2
TARGET_IMAGE = "LTLB03.jpg" #adjust here to the image you want to evaluate performance on (e.g., "LTLB03.jpg")
OUT_DIR      = "~/Segmentation_Evaluation/"

# ── Build task_id → image_name mapping ────────────────────────────────
with open(JSON_PATH, "r") as f:
    tasks = json.load(f)

TASK_IDS = [1, 2, 3, 4, 6, 7, 10, 11, 14, 15, 21, 24] # <--- adjust (this is for rater 1!)
#TASK_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] # <--- (this is for rater 2!)

task_to_image = {}
for task_id, task in zip(TASK_IDS, tasks):
    raw_path = task["image"]
    image_name = raw_path.split("-", 1)[1]
    task_to_image[task_id] = image_name

# ── Load predictions ───────────────────────────────────────────────────
with open(PICKLE_PATH, "rb") as f:
    df = pickle.load(f)
predictions = {row["img_name"]: np.array(row["predicted_segmentation"])
               for _, row in df.iterrows()}

def parse_class_id_and_name(filename):
    task_m = re.search(r'task-(\d+)', filename)
    task_id = int(task_m.group(1)) if task_m else None
    m = re.search(r'"(\d+)":\s*"([^"]+)"', filename)
    if m:
        return task_id, int(m.group(1)), m.group(2)
    m = re.search(r'tag-(\d+)_(\w+)', filename)
    if m:
        return task_id, int(m.group(1)), m.group(2)
    return task_id, None, None

GREENERY_IDS = {4, 9, 17}
BUILT_IDS    = {1, 6, 11, 84}
SOLO_IDS     = {2: "sky", 12: "person", 20: "car"}
PAPER_IDS    = GREENERY_IDS | BUILT_IDS | set(SOLO_IDS)

def compute_iou(gt_binary, pred_binary):
    intersection = np.logical_and(gt_binary, pred_binary).sum()
    union        = np.logical_or(gt_binary, pred_binary).sum()
    return intersection / union if union > 0 else float("nan"), gt_binary, pred_binary

def resize_gt(gt_mask, shape):
    H, W = shape
    return np.array(Image.fromarray(gt_mask).resize((W, H), Image.NEAREST)) > 0

# ── Run ────────────────────────────────────────────────────────────────
pred_mask = predictions[TARGET_IMAGE]
results   = []
masks     = {}
greenery_gt = None
built_gt    = None

for fname in os.listdir(NPY_DIR):
    if not fname.endswith(".npy"):
        continue
    task_id, class_id, class_name = parse_class_id_and_name(fname)
    if task_id is None or class_id is None:
        continue
    if task_to_image.get(task_id) != TARGET_IMAGE:
        continue
    if class_id not in PAPER_IDS:
        continue
    gt_mask = np.load(os.path.join(NPY_DIR, fname))
    gt_bin  = resize_gt(gt_mask, pred_mask.shape)
    if class_id in GREENERY_IDS:
        greenery_gt = gt_bin if greenery_gt is None else greenery_gt | gt_bin
    elif class_id in BUILT_IDS:
        built_gt = gt_bin if built_gt is None else built_gt | gt_bin
    else:
        pred_bin = (pred_mask == class_id)
        iou, gt_b, pred_b = compute_iou(gt_bin, pred_bin)
        results.append({"class": class_id, "name": class_name, "iou": iou})
        masks[class_id] = {"name": class_name, "gt": gt_b, "pred": pred_b, "iou": iou}
        print(f"class {class_id:3d} ({class_name:15s}) | IoU: {iou:.4f}")

# ── Combined greenery ──────────────────────────────────────────────────
if greenery_gt is not None:
    pred_bin = np.isin(pred_mask, list(GREENERY_IDS))
    iou, gt_b, pred_b = compute_iou(greenery_gt, pred_bin)
    results.append({"class": "greenery", "name": "greenery (4+9+17)", "iou": iou})
    masks["greenery"] = {"name": "greenery (4+9+17)", "gt": gt_b, "pred": pred_b, "iou": iou}
    print(f"greenery (4+9+17)              | IoU: {iou:.4f}")

# ── Combined built ─────────────────────────────────────────────────────
if built_gt is not None:
    pred_bin = np.isin(pred_mask, list(BUILT_IDS))
    iou, gt_b, pred_b = compute_iou(built_gt, pred_bin)
    results.append({"class": "built", "name": "built (1+6+11+84)", "iou": iou})
    masks["built"] = {"name": "built (1+6+11+84)", "gt": gt_b, "pred": pred_b, "iou": iou}
    print(f"built (1+6+11+84)              | IoU: {iou:.4f}")

results_df = pd.DataFrame(results).sort_values("iou", ascending=False)
print(f"\nMean IoU: {results_df['iou'].mean():.4f}")

# ── Plot 1: IoU bar chart ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
colors = ["#2ecc71" if v > 0.5 else "#e67e22" if v > 0.25 else "#e74c3c"
          for v in results_df["iou"]]
bars = ax.barh(
    results_df["name"],
    results_df["iou"],
    color=colors
)
ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="IoU=0.5")
ax.set_xlabel("IoU")
ax.set_title(f"Per-class IoU — {TARGET_IMAGE}")
ax.set_xlim(0, 1)
for bar, val in zip(bars, results_df["iou"]):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9)
patches = [
    mpatches.Patch(color="#2ecc71", label="IoU > 0.5 (good)"),
    mpatches.Patch(color="#e67e22", label="IoU 0.25–0.5 (ok)"),
    mpatches.Patch(color="#e74c3c", label="IoU < 0.25 (poor)"),
]
ax.legend(handles=patches, loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"iou_barchart_{os.path.splitext(TARGET_IMAGE)[0]}.png"), dpi=150)
plt.close()
print("Saved iou_barchart.png")

# ── Plot 2: per-class GT vs Pred overlay grid ──────────────────────────
classes_sorted = sorted(masks.keys(), key=str)
n = len(classes_sorted)
fig, axes = plt.subplots(n, 3, figsize=(12, 3 * n))
if n == 1:
    axes = [axes]

for i, cid in enumerate(classes_sorted):
    m      = masks[cid]
    gt     = m["gt"].astype(np.uint8)
    pred   = m["pred"].astype(np.uint8)
    overlap = gt + pred * 2  # 0=bg, 1=GT only, 2=pred only, 3=both

    axes[i][0].imshow(gt,      cmap="gray");   axes[i][0].set_title(f"GT — {m['name']}")
    axes[i][1].imshow(pred,    cmap="gray");   axes[i][1].set_title(f"Pred — {m['name']}")
    axes[i][2].imshow(overlap, cmap="viridis"); axes[i][2].set_title(f"Overlap | IoU={m['iou']:.3f}")
    for ax in axes[i]:
        ax.axis("off")

plt.suptitle(f"GT vs Prediction — {TARGET_IMAGE}", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"iou_grid_{os.path.splitext(TARGET_IMAGE)[0]}.png"), dpi=120, bbox_inches="tight")
plt.close()
print("Saved iou_grid.png")