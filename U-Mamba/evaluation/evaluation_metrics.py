import os
import nibabel as nib
import numpy as np
from sklearn.metrics import confusion_matrix


pred_dir = "/content/drive/MyDrive/topcow_data/Inference/ACC/CT/CT_16"
gt_dir = "/content/U-Mamba/data/nnUNet_raw/Dataset993_TopCoW/labelsTs"   # folder with .nii ground truth

def load_nifti(path):
    return nib.load(path).get_fdata().astype(np.uint8)

def compute_metrics(pred, gt):
    """Compute metrics for a single case."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    tp = intersection
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()
    tn = np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum()

    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    iou = tp / (union + 1e-8)
    sensitivity = tp / (tp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    vol_pred = pred.sum()
    vol_gt = gt.sum()
    volume_similarity = 1 - abs(vol_pred - vol_gt) / (vol_pred + vol_gt + 1e-8)

    return dice, iou, sensitivity, precision, specificity, volume_similarity

# Collect metrics per case
results = []
for filename in os.listdir(pred_dir):
    if not filename.endswith(".nii.gz"):
        continue

    pred_path = os.path.join(pred_dir, filename)
    gt_path = os.path.join(gt_dir, filename)
     # adjust if your GT is .nii.gz too
    if not os.path.exists(gt_path):
        print(f"No ground truth for {filename}")
        continue

    pred = load_nifti(pred_path)
    gt = load_nifti(gt_path)

    metrics = compute_metrics(pred, gt)
    results.append(metrics)
    print(f"{filename}: Dice={metrics[0]:.4f}, IoU={metrics[1]:.4f}, Sens={metrics[2]:.4f}, "
          f"Prec={metrics[3]:.4f}, Spec={metrics[4]:.4f}, VS={metrics[5]:.4f}")

# Compute mean across all cases
if results:
    results = np.array(results)
    mean_metrics = results.mean(axis=0)
    print("\n=== Mean Metrics Across All Cases ===")
    print(f"Mean Dice: {mean_metrics[0]:.4f}")
    print(f"Mean IoU: {mean_metrics[1]:.4f}")
    print(f"Mean Sensitivity: {mean_metrics[2]:.4f}")
    print(f"Mean Precision: {mean_metrics[3]:.4f}")
    print(f"Mean Specificity: {mean_metrics[4]:.4f}")
    print(f"Mean Volume Similarity: {mean_metrics[5]:.4f}")