"""
Evaluate inference results across multiple experiment subfolders.

Each experiment subfolder (e.g. 48, 53, ...) contains an 'inference' folder
with nnU-Net named predictions. Uses a mapping file to match predictions
to GT files with original names.

Usage:
  python evaluate_all_experiments.py <experiments_dir> <gt_dir> <mapping_csv>

Example:
  python evaluate_all_experiments.py /path/to/results /path/to/gt_folder /path/to/file_mapping.csv

Output:
  - summary_results.csv: mean+std metrics per experiment (one row liver, one row tumor)
  - per-case CSVs inside each experiment folder
"""

import os
import sys
import csv
import statistics
import numpy as np
import nibabel as nib
from pathlib import Path
from datetime import datetime
from scipy.ndimage import distance_transform_edt
from skimage.measure import label, regionprops, regionprops_table


EXPERIMENT_FOLDERS = [48, 53, 58, 63, 68, 73, 79, 86, 91, 97, 102, 108, 113]


# ============================================================
# METRICS FUNCTIONS
# ============================================================

def calculating_dice(gt, predictions):
    predictions = predictions.astype(np.uint8)
    gt = gt.astype(np.uint8)

    if not predictions.shape == gt.shape:
        raise ValueError(
            "Predictions have shape {} which do not match ground truth shape of {}".format(
                predictions.shape, gt.shape
            )
        )

    try:
        tk_pd = np.greater(predictions, 0)
        tk_gt = np.greater(gt, 0)
        tk_dice = 2 * np.logical_and(tk_pd, tk_gt).sum() / (
            tk_pd.sum() + tk_gt.sum()
        )
    except ZeroDivisionError:
        return 0.0, 0.0

    try:
        tu_pd = np.greater(predictions, 1)
        tu_gt = np.greater(gt, 1)
        if (tu_pd.sum() + tu_gt.sum()) == 0:
            return tk_dice, 0.0
        else:
            tu_dice = 2 * np.logical_and(tu_pd, tu_gt).sum() / (
                tu_pd.sum() + tu_gt.sum()
            )
    except ZeroDivisionError:
        return tk_dice, 0.0

    return tk_dice, tu_dice


def _get_surface(mask):
    """Extract surface voxels using binary erosion (faster than EDT)."""
    from scipy.ndimage import binary_erosion
    eroded = binary_erosion(mask)
    return mask & ~eroded


def _compute_surface_metrics(gt_mask, pred_mask, voxel_spacing, tolerance_mm=1.5, percentile=95):
    """
    Compute HD, HD95, and NSD in a single pass.
    Only 2 EDT calls instead of 8.
    Returns (hd, hd95, nsd).
    """
    if gt_mask.sum() == 0 or pred_mask.sum() == 0:
        return np.inf, np.inf, 0.0

    # Surface extraction via erosion (fast)
    gt_surface = _get_surface(gt_mask)
    pred_surface = _get_surface(pred_mask)

    n_gt_surface = gt_surface.sum()
    n_pred_surface = pred_surface.sum()

    if n_gt_surface == 0 or n_pred_surface == 0:
        return np.inf, np.inf, 0.0

    # Only 2 EDT calls total (with voxel spacing for physical distances)
    dist_from_gt = distance_transform_edt(~gt_mask, sampling=voxel_spacing)
    dist_from_pred = distance_transform_edt(~pred_mask, sampling=voxel_spacing)

    dist_pred_to_gt = dist_from_gt[pred_surface]
    dist_gt_to_pred = dist_from_pred[gt_surface]

    all_distances = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])

    # HD and HD95
    hd = float(np.max(all_distances))
    hd95 = float(np.percentile(all_distances, percentile))

    # NSD
    pred_to_gt_within = (dist_pred_to_gt <= tolerance_mm).sum()
    gt_to_pred_within = (dist_gt_to_pred <= tolerance_mm).sum()
    nsd = float((pred_to_gt_within + gt_to_pred_within) / (n_pred_surface + n_gt_surface))

    return hd, hd95, nsd


def calculating_surface_metrics(gt, predictions, voxel_spacing, tolerance_mm=1.5):
    """
    Compute HD, HD95, NSD for liver+tumor and tumor in one pass.
    Returns (tk_hd, tk_hd95, tk_nsd, tu_hd, tu_hd95, tu_nsd).
    """
    predictions = predictions.astype(np.uint8)
    gt = gt.astype(np.uint8)

    tk_pd = np.greater(predictions, 0)
    tk_gt = np.greater(gt, 0)
    tk_hd, tk_hd95, tk_nsd = _compute_surface_metrics(tk_gt, tk_pd, voxel_spacing, tolerance_mm)

    tu_pd = np.greater(predictions, 1)
    tu_gt = np.greater(gt, 1)
    tu_hd, tu_hd95, tu_nsd = _compute_surface_metrics(tu_gt, tu_pd, voxel_spacing, tolerance_mm)

    return tk_hd, tk_hd95, tk_nsd, tu_hd, tu_hd95, tu_nsd


def get_max_diameter_cm(binary_mask, voxel_spacing):
    """
    Compute the maximum physical diameter (in cm) of a binary lesion mask.
    Uses the bounding box extents scaled by voxel spacing.
    voxel_spacing: tuple of (row_spacing, col_spacing, slice_thickness) in mm.
    """
    coords = np.argwhere(binary_mask)
    if len(coords) == 0:
        return 0.0
    # Physical extent along each axis
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    extents_mm = (maxs - mins + 1) * np.array(voxel_spacing)
    # Max diameter = diagonal of the bounding box
    max_diameter_mm = np.sqrt(np.sum(extents_mm ** 2))
    return max_diameter_mm / 10.0  # mm -> cm


def get_size_group(diameter_cm):
    """Classify lesion diameter into size groups."""
    if diameter_cm < 2.0:
        return "<2cm"
    elif diameter_cm < 3.0:
        return "2-3cm"
    elif diameter_cm < 4.0:
        return "3-4cm"
    else:
        return "4-8cm"


SIZE_GROUPS = ["<2cm", "2-3cm", "3-4cm", "4-8cm"]


def lesion_calc(gt, prediction, case_name, file_name1, file_name2, voxel_spacing=None):
    """
    Lesion detection: class 2 = tumor.
    Returns (n_gt_lesions, n_pred_lesions, n_correct, lesion_sizes_dices).
    lesion_sizes_dices: list of (size_group, dice) per GT lesion.
    """
    segmentation = (prediction == 2)
    labels_pred = label(segmentation)
    props1 = regionprops(labels_pred)

    gt_ = (gt == 2)
    gt_labels = label(gt_)
    props2 = regionprops(gt_labels)
    pro2 = regionprops_table(gt_labels, gt_labels,
                             properties=['label', 'bbox', 'area'])

    file_name1 = Path(file_name1)
    file_name2 = Path(file_name2)

    if not file_name2.exists():
        with open(file_name2, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter='|')
            writer.writerow(['case', 'tumors_area', 'found', 'dice', 'diameter_cm', 'size_group'])

    count = 0
    lesion_sizes_dices = []

    if len(props2) > 0:
        for i in range(len(props2)):
            vol_pred = segmentation[
                pro2['bbox-0'][i]:pro2['bbox-3'][i],
                pro2['bbox-1'][i]:pro2['bbox-4'][i],
                pro2['bbox-2'][i]:pro2['bbox-5'][i]
            ].astype(int)
            vol_gt = gt[
                pro2['bbox-0'][i]:pro2['bbox-3'][i],
                pro2['bbox-1'][i]:pro2['bbox-4'][i],
                pro2['bbox-2'][i]:pro2['bbox-5'][i]
            ].copy()
            vol_gt[vol_gt < 2] = 0
            vol_gt[vol_gt == 2] = 1
            perc = vol_pred.sum() / vol_gt.sum() if vol_gt.sum() > 0 else 0

            if perc >= 0.1:
                count += 1
                temp_count = 1
            else:
                temp_count = 0

            tumor_dice, _ = calculating_dice(vol_gt, vol_pred)

            # Compute physical size
            if voxel_spacing is not None:
                diameter_cm = get_max_diameter_cm(vol_gt, voxel_spacing)
            else:
                diameter_cm = 0.0
            size_group = get_size_group(diameter_cm)
            lesion_sizes_dices.append((size_group, float(tumor_dice)))

            with open(file_name2, 'a+', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter='|')
                writer.writerow([case_name, pro2['area'][i], temp_count,
                                 round(tumor_dice, 3), round(diameter_cm, 2), size_group])

    if not file_name1.exists():
        with open(file_name1, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter='|')
            writer.writerow(['case', 'tumors_gt', 'tumors_prediction', 'correct'])

    with open(file_name1, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='|')
        writer.writerow([case_name, len(props2), len(props1), count])

    return len(props2), len(props1), count, lesion_sizes_dices


# ============================================================
# MAPPING
# ============================================================

def load_mapping(mapping_csv):
    """Load mapping: nnunet_name -> original_name."""
    mapping = {}
    with open(mapping_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["nnunet_name"]] = row["original_name"]
    return mapping


# ============================================================
# EVALUATION
# ============================================================

def evaluate_experiment(exp_name, inference_dir, gt_dir, mapping, output_dir):
    """Evaluate one experiment folder. Returns dict with mean metrics."""

    pred_files = sorted([f for f in os.listdir(inference_dir) if f.endswith(".nii.gz")])

    if not pred_files:
        print(f"  No predictions found in {inference_dir}")
        return None

    # Metric accumulators
    tk_dice_list, tu_dice_list = [], []
    sn_liver, pr_liver, rc_liver, voe_liver = [], [], [], []
    sn_tumor, pr_tumor, rc_tumor, voe_tumor = [], [], [], []
    hd_liver, hd95_liver, hd_tumor, hd95_tumor = [], [], [], []
    nsd_liver, nsd_tumor = [], []
    count_gt, count_prediction, count_correct = 0, 0, 0
    # Dice per lesion size group
    size_group_dices = {g: [] for g in SIZE_GROUPS}

    # Per-case CSV
    case_csv = os.path.join(output_dir, f"cases_{exp_name}.csv")
    with open(case_csv, "w", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerow(["case", "object", "sn", "pr", "rc", "dice", "voe", "hd", "hd95", "nsd"])

    # Build list of available GT files for direct matching
    gt_files = set(f for f in os.listdir(gt_dir) if f.endswith(".nii.gz")) if os.path.exists(gt_dir) else set()

    for pred_file in pred_files:
        gt_name = None

        if mapping:
            # Try mapping: pred_file -> original name
            if pred_file in mapping:
                gt_name = mapping[pred_file]
            else:
                pred_with_suffix = pred_file.replace(".nii.gz", "_0000.nii.gz")
                if pred_with_suffix in mapping:
                    gt_name = mapping[pred_with_suffix]

        if gt_name is None:
            # No mapping or not found in mapping -> try direct match
            if pred_file in gt_files:
                gt_name = pred_file
            else:
                # Try without _0000 suffix
                no_suffix = pred_file.replace("_0000.nii.gz", ".nii.gz")
                if no_suffix in gt_files:
                    gt_name = no_suffix
                else:
                    print(f"  WARNING: {pred_file} - no matching GT found, skipping")
                    continue

        gt_path = os.path.join(gt_dir, gt_name)
        pred_path = os.path.join(inference_dir, pred_file)

        if not os.path.exists(gt_path):
            print(f"  WARNING: GT file not found: {gt_path}, skipping")
            continue

        # Load data
        gt_img = nib.load(gt_path)
        prediction = nib.load(pred_path).get_fdata()
        gt = gt_img.get_fdata()
        voxel_spacing = gt_img.header.get_zooms()[:3]  # (row, col, slice) in mm
        gt = np.round(gt)
        prediction = np.round(prediction)

        if prediction.shape != gt.shape:
            print(f"  WARNING: Shape mismatch for {pred_file}: pred {prediction.shape} vs gt {gt.shape}, skipping")
            continue

        print(f"  {pred_file} -> {gt_name}")

        # Dice
        tk, tu = calculating_dice(gt, prediction)
        tk_dice_list.append(tk)
        tu_dice_list.append(tu)

        # Surface metrics: HD, HD95, NSD in one pass
        tk_h, tk_h95, tk_nsd, tu_h, tu_h95, tu_nsd = calculating_surface_metrics(
            gt, prediction, voxel_spacing, tolerance_mm=1.5
        )
        hd_liver.append(tk_h)
        hd95_liver.append(tk_h95)
        nsd_liver.append(tk_nsd)
        hd_tumor.append(tu_h)
        hd95_tumor.append(tu_h95)
        nsd_tumor.append(tu_nsd)

        # Liver metrics (class 1)
        prediction_uint8 = prediction.astype(np.uint8)
        gt_uint8 = gt.astype(np.uint8)

        gt_p = np.greater(gt_uint8, 0)
        pos_pred = np.greater(prediction_uint8, 0)
        neg_pred = np.less_equal(prediction_uint8, 0)
        fp = np.logical_and(pos_pred, ~gt_p).sum()
        fn = np.logical_and(neg_pred, gt_p).sum()
        tp = np.logical_and(pos_pred, gt_p).sum()
        voe_l = 100 * (pos_pred.sum() - gt_p.sum()) / gt_p.sum() if gt_p.sum() > 0 else 0
        sn_l = float(tp / (tp + fn + 1))
        pr_l = float(tp / (tp + fp + 1))
        rc_l = float(tp / (tp + fn + 1))

        sn_liver.append(sn_l)
        pr_liver.append(pr_l)
        rc_liver.append(rc_l)
        voe_liver.append(voe_l)

        # Tumor metrics (class 2)
        gt_p = np.greater(gt_uint8, 1)
        pos_pred = np.greater(prediction_uint8, 1)
        neg_pred = np.less_equal(prediction_uint8, 1)
        fp = np.logical_and(pos_pred, ~gt_p).sum()
        fn = np.logical_and(neg_pred, gt_p).sum()
        tp = np.logical_and(pos_pred, gt_p).sum()
        voe_t = 100 * (pos_pred.sum() - gt_p.sum()) / gt_p.sum() if gt_p.sum() > 0 else 0
        sn_t = float(tp / (tp + fn + 1))
        pr_t = float(tp / (tp + fp + 1))
        rc_t = float(tp / (tp + fn + 1))

        sn_tumor.append(sn_t)
        pr_tumor.append(pr_t)
        rc_tumor.append(rc_t)
        voe_tumor.append(voe_t)

        # Write per-case
        with open(case_csv, "a", newline="") as f:
            writer = csv.writer(f, delimiter="|")
            writer.writerow([gt_name, "liver", round(sn_l, 3), round(pr_l, 3), round(rc_l, 3), round(tk, 3), round(voe_l, 3), round(tk_h, 3), round(tk_h95, 3), round(tk_nsd, 3)])
            writer.writerow([gt_name, "tumor", round(sn_t, 3), round(pr_t, 3), round(rc_t, 3), round(tu, 3), round(voe_t, 3), round(tu_h, 3), round(tu_h95, 3), round(tu_nsd, 3)])

        # Lesion detection
        n_gt, n_pred, n_corr, lesion_sd = lesion_calc(
            gt, prediction, gt_name,
            Path(output_dir) / f"lesions_{exp_name}.csv",
            Path(output_dir) / f"lesion_sizes_{exp_name}.csv",
            voxel_spacing=voxel_spacing,
        )
        count_gt += n_gt
        count_prediction += n_pred
        count_correct += n_corr
        for size_group, dice_val in lesion_sd:
            size_group_dices[size_group].append(dice_val)

    if not tk_dice_list:
        return None

    # Filter out inf values for hausdorff means
    hd_liver_clean = [x for x in hd_liver if x != np.inf]
    hd95_liver_clean = [x for x in hd95_liver if x != np.inf]
    hd_tumor_clean = [x for x in hd_tumor if x != np.inf]
    hd95_tumor_clean = [x for x in hd95_tumor if x != np.inf]

    def safe_std(lst):
        return statistics.stdev(lst) if len(lst) > 1 else 0.0

    results = {
        "exp": exp_name,
        "n_cases": len(tk_dice_list),
        "liver_dice": statistics.mean(tk_dice_list),
        "liver_dice_std": safe_std(tk_dice_list),
        "liver_sn": statistics.mean(sn_liver),
        "liver_sn_std": safe_std(sn_liver),
        "liver_pr": statistics.mean(pr_liver),
        "liver_pr_std": safe_std(pr_liver),
        "liver_rc": statistics.mean(rc_liver),
        "liver_rc_std": safe_std(rc_liver),
        "liver_voe": statistics.mean(voe_liver),
        "liver_voe_std": safe_std(voe_liver),
        "liver_hd": statistics.mean(hd_liver_clean) if hd_liver_clean else np.inf,
        "liver_hd_std": safe_std(hd_liver_clean) if len(hd_liver_clean) > 1 else 0.0,
        "liver_hd95": statistics.mean(hd95_liver_clean) if hd95_liver_clean else np.inf,
        "liver_hd95_std": safe_std(hd95_liver_clean) if len(hd95_liver_clean) > 1 else 0.0,
        "liver_nsd": statistics.mean(nsd_liver),
        "liver_nsd_std": safe_std(nsd_liver),
        "tumor_dice": statistics.mean(tu_dice_list),
        "tumor_dice_std": safe_std(tu_dice_list),
        "tumor_sn": statistics.mean(sn_tumor),
        "tumor_sn_std": safe_std(sn_tumor),
        "tumor_pr": statistics.mean(pr_tumor),
        "tumor_pr_std": safe_std(pr_tumor),
        "tumor_rc": statistics.mean(rc_tumor),
        "tumor_rc_std": safe_std(rc_tumor),
        "tumor_voe": statistics.mean(voe_tumor),
        "tumor_voe_std": safe_std(voe_tumor),
        "tumor_hd": statistics.mean(hd_tumor_clean) if hd_tumor_clean else np.inf,
        "tumor_hd_std": safe_std(hd_tumor_clean) if len(hd_tumor_clean) > 1 else 0.0,
        "tumor_hd95": statistics.mean(hd95_tumor_clean) if hd95_tumor_clean else np.inf,
        "tumor_hd95_std": safe_std(hd95_tumor_clean) if len(hd95_tumor_clean) > 1 else 0.0,
        "tumor_nsd": statistics.mean(nsd_tumor),
        "tumor_nsd_std": safe_std(nsd_tumor),
        "lesion_gt": count_gt,
        "lesion_pred": count_prediction,
        "lesion_correct": count_correct,
    }

    # Add size-grouped lesion dice
    for g in SIZE_GROUPS:
        dices = size_group_dices[g]
        results[f"lesion_dice_{g}"] = statistics.mean(dices) if dices else 0.0
        results[f"lesion_dice_{g}_std"] = safe_std(dices) if len(dices) > 1 else 0.0
        results[f"lesion_dice_{g}_n"] = len(dices)

    print(f"  -> Liver Dice: {results['liver_dice']:.3f}, Tumor Dice: {results['tumor_dice']:.3f}")
    for g in SIZE_GROUPS:
        n = results[f"lesion_dice_{g}_n"]
        if n > 0:
            print(f"     Lesion {g}: Dice {results[f'lesion_dice_{g}']:.3f}+/-{results[f'lesion_dice_{g}_std']:.3f} (n={n})")
    return results


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python evaluate_all_experiments.py <experiments_dir> <gt_dir>")
        print("  python evaluate_all_experiments.py <experiments_dir> <gt_dir> <mapping_csv>")
        print("")
        print("  experiments_dir: folder containing subfolders 48, 53, ..., 113")
        print("  gt_dir:          folder with GT segmentation files")
        print("  mapping_csv:     (optional) file_mapping.csv from convert_to_nnunet_predict.py")
        sys.exit(1)

    experiments_dir = sys.argv[1]
    gt_dir = sys.argv[2]
    mapping_csv = sys.argv[3] if len(sys.argv) >= 4 else None

    if not os.path.exists(experiments_dir):
        print(f"ERROR: Experiments directory not found: {experiments_dir}")
        sys.exit(1)
    if not os.path.exists(gt_dir):
        print(f"ERROR: GT directory not found: {gt_dir}")
        sys.exit(1)

    mapping = {}
    if mapping_csv:
        if not os.path.exists(mapping_csv):
            print(f"ERROR: Mapping file not found: {mapping_csv}")
            sys.exit(1)
        mapping = load_mapping(mapping_csv)
        print(f"Loaded mapping with {len(mapping)} entries")
    else:
        print("No mapping file provided - will match files by name")

    gt_folder_name = os.path.basename(os.path.normpath(gt_dir))
    print(f"GT folder: {gt_folder_name}")

    # Summary CSV
    summary_path = os.path.join(experiments_dir, "summary_results.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerow([
            "experiment", "gt_folder", "datetime", "n_cases", "object",
            "sn", "sn_std",
            "pr", "pr_std", "rc", "rc_std", "dice", "dice_std",
            "voe", "voe_std", "hd", "hd_std", "hd95", "hd95_std",
            "nsd_1.5mm", "nsd_1.5mm_std",
            "lesion_gt", "lesion_pred", "lesion_correct",
            "dice_<2cm", "dice_<2cm_std", "n_<2cm",
            "dice_2-3cm", "dice_2-3cm_std", "n_2-3cm",
            "dice_3-4cm", "dice_3-4cm_std", "n_3-4cm",
            "dice_4-8cm", "dice_4-8cm_std", "n_4-8cm",
        ])

    all_results = []

    for exp_id in EXPERIMENT_FOLDERS:
        exp_folder = os.path.join(experiments_dir, str(exp_id))
        inference_dir = os.path.join(exp_folder, "inference")

        if not os.path.exists(inference_dir):
            print(f"\n[{exp_id}] inference folder not found at {inference_dir}, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating experiment {exp_id}")
        print(f"{'='*60}")

        output_dir = exp_folder
        results = evaluate_experiment(str(exp_id), inference_dir, gt_dir, mapping, output_dir)

        if results is None:
            print(f"  No valid results for experiment {exp_id}")
            continue

        all_results.append(results)

        # Append to summary
        with open(summary_path, "a", newline="") as f:
            writer = csv.writer(f, delimiter="|")
            r = results
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([
                r["exp"], gt_folder_name, now, r["n_cases"], "liver",
                round(r["liver_sn"], 3), round(r["liver_sn_std"], 3),
                round(r["liver_pr"], 3), round(r["liver_pr_std"], 3),
                round(r["liver_rc"], 3), round(r["liver_rc_std"], 3),
                round(r["liver_dice"], 3), round(r["liver_dice_std"], 3),
                round(r["liver_voe"], 3), round(r["liver_voe_std"], 3),
                round(r["liver_hd"], 3), round(r["liver_hd_std"], 3),
                round(r["liver_hd95"], 3), round(r["liver_hd95_std"], 3),
                round(r["liver_nsd"], 3), round(r["liver_nsd_std"], 3),
                r["lesion_gt"], r["lesion_pred"], r["lesion_correct"],
                "", "", "", "", "", "", "", "", "", "", "", "",
            ])
            size_cols = []
            for g in SIZE_GROUPS:
                size_cols.extend([
                    round(r[f"lesion_dice_{g}"], 3),
                    round(r[f"lesion_dice_{g}_std"], 3),
                    r[f"lesion_dice_{g}_n"],
                ])
            writer.writerow([
                r["exp"], gt_folder_name, now, r["n_cases"], "tumor",
                round(r["tumor_sn"], 3), round(r["tumor_sn_std"], 3),
                round(r["tumor_pr"], 3), round(r["tumor_pr_std"], 3),
                round(r["tumor_rc"], 3), round(r["tumor_rc_std"], 3),
                round(r["tumor_dice"], 3), round(r["tumor_dice_std"], 3),
                round(r["tumor_voe"], 3), round(r["tumor_voe_std"], 3),
                round(r["tumor_hd"], 3), round(r["tumor_hd_std"], 3),
                round(r["tumor_hd95"], 3), round(r["tumor_hd95_std"], 3),
                round(r["tumor_nsd"], 3), round(r["tumor_nsd_std"], 3),
                r["lesion_gt"], r["lesion_pred"], r["lesion_correct"],
            ] + size_cols)

    # Print final summary table
    if all_results:
        print(f"\n{'='*120}")
        print("SUMMARY")
        print(f"{'='*120}")
        print(f"{'Exp':>6} | {'Liver Dice':>18} | {'Tumor Dice':>18} | {'Liver HD95':>18} | {'Tumor HD95':>18} | {'Liver NSD':>18} | {'Tumor NSD':>18} | {'Lesions':>10}")
        print("-" * 140)
        for r in all_results:
            ld = f"{r['liver_dice']:.3f}+/-{r['liver_dice_std']:.3f}"
            td = f"{r['tumor_dice']:.3f}+/-{r['tumor_dice_std']:.3f}"
            lh = f"{r['liver_hd95']:.1f}+/-{r['liver_hd95_std']:.1f}" if r['liver_hd95'] != np.inf else "inf"
            th = f"{r['tumor_hd95']:.1f}+/-{r['tumor_hd95_std']:.1f}" if r['tumor_hd95'] != np.inf else "inf"
            ln = f"{r['liver_nsd']:.3f}+/-{r['liver_nsd_std']:.3f}"
            tn = f"{r['tumor_nsd']:.3f}+/-{r['tumor_nsd_std']:.3f}"
            print(f"{r['exp']:>6} | {ld:>18} | {td:>18} | {lh:>18} | {th:>18} | {ln:>18} | {tn:>18} | {r['lesion_correct']}/{r['lesion_gt']}")

        # Lesion Dice by size group
        print(f"\n{'='*120}")
        print("LESION DICE BY SIZE GROUP")
        print(f"{'='*120}")
        print(f"{'Exp':>6} | {'<2cm':>22} | {'2-3cm':>22} | {'3-4cm':>22} | {'4-8cm':>22}")
        print("-" * 120)
        for r in all_results:
            cols = []
            for g in SIZE_GROUPS:
                n = r[f"lesion_dice_{g}_n"]
                if n > 0:
                    cols.append(f"{r[f'lesion_dice_{g}']:.3f}+/-{r[f'lesion_dice_{g}_std']:.3f} (n={n})")
                else:
                    cols.append("- (n=0)")
            print(f"{r['exp']:>6} | {cols[0]:>22} | {cols[1]:>22} | {cols[2]:>22} | {cols[3]:>22}")

        print(f"\nFull results saved to: {summary_path}")


if __name__ == "__main__":
    main()
