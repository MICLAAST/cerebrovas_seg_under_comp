import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

# ==== Paths ====
pred_dir = "/content/drive/MyDrive/topcow_data/Inference/ACC/CT/CT_16"
gt_dir = "/content/U-Mamba/data/nnUNet_raw/Dataset993_TopCoW/labelsTs"
save_dir = "/content/drive/MyDrive/topcow_data/Visualizations/Acc/CT/CT_16"
os.makedirs(save_dir, exist_ok=True)

# ==== Function to plot GT + Prediction side by side ====
def plot_surface_side(gt_data, pred_data, case_id, factor=2):
    gt_ds = gt_data[::factor, ::factor, ::factor]
    pred_ds = pred_data[::factor, ::factor, ::factor]

    # Skip empty predictions
    if np.max(pred_ds) == 0 and np.max(gt_ds) == 0:
        print(f"⚠️ Both empty: {case_id}")
        return

    verts_gt, faces_gt, _, _ = measure.marching_cubes(gt_ds, level=0.5)
    verts_pred, faces_pred, _, _ = measure.marching_cubes(pred_ds, level=0.5)

    fig = plt.figure(figsize=(16, 8))

    # --- Ground Truth ---
    ax1 = fig.add_subplot(121, projection="3d")
    mesh_gt = Poly3DCollection(verts_gt[faces_gt], alpha=0.5, facecolor="green")
    ax1.add_collection3d(mesh_gt)
    ax1.set_xlim(0, gt_ds.shape[0])
    ax1.set_ylim(0, gt_ds.shape[1])
    ax1.set_zlim(0, gt_ds.shape[2])
    ax1.set_title(f"Ground Truth\n{case_id}")

    # --- Prediction ---
    ax2 = fig.add_subplot(122, projection="3d")
    mesh_pred = Poly3DCollection(verts_pred[faces_pred], alpha=0.5, facecolor="red")
    ax2.add_collection3d(mesh_pred)
    ax2.set_xlim(0, pred_ds.shape[0])
    ax2.set_ylim(0, pred_ds.shape[1])
    ax2.set_zlim(0, pred_ds.shape[2])
    ax2.set_title(f"Prediction\n{case_id}")

    save_path = os.path.join(save_dir, f"{case_id}_3D.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Saved: {save_path}")

# ==== Main Loop (limit to 5 cases) ====
processed = 0
for fname in sorted(os.listdir(gt_dir)):
    if fname.endswith(".nii") or fname.endswith(".nii.gz"):
        case_id = os.path.splitext(fname)[0]
        gt_path = os.path.join(gt_dir, fname)
        pred_path = os.path.join(pred_dir, fname)

        if not os.path.exists(pred_path):
            print(f"❌ Prediction not found for {case_id}")
            continue

        gt_data = nib.load(gt_path).get_fdata().astype(np.uint8)
        pred_data = nib.load(pred_path).get_fdata().astype(np.uint8)

        plot_surface_side(gt_data, pred_data, case_id)
        processed += 1

        if processed >= 5:
            print("\n✅ Done! Visualized first 5 matching cases.")
            break