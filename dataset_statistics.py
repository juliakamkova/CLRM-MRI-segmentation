"""
Extract dataset statistics from a folder of NIfTI files for paper description.

Computes acquisition parameters, volume statistics, and lesion analysis.

Usage:
  Images only:
    python dataset_statistics.py <image_dir>

  Images + labels:
    python dataset_statistics.py <image_dir> <label_dir>

  Images + labels + mapping:
    python dataset_statistics.py <image_dir> <label_dir> <mapping_csv>

Output:
  - dataset_statistics.csv: per-volume stats
  - dataset_summary.txt: summary for paper (copy-paste ready)
"""

import os
import sys
import csv
import numpy as np
import nibabel as nib
from collections import defaultdict


def get_size_group(diameter_cm):
    if diameter_cm < 2.0:
        return "<2cm"
    elif diameter_cm < 3.0:
        return "2-3cm"
    elif diameter_cm < 4.0:
        return "3-4cm"
    else:
        return "4-8cm"


def analyze_volume(img_path, label_path=None):
    """Extract all stats from one volume."""
    img_nii = nib.load(img_path)
    header = img_nii.header
    data = img_nii.get_fdata()

    zooms = header.get_zooms()
    voxel_spacing = zooms[:3]  # mm

    stats = {
        "filename": os.path.basename(img_path),
        "shape_x": data.shape[0],
        "shape_y": data.shape[1],
        "shape_z": data.shape[2] if len(data.shape) > 2 else 1,
        "spacing_x": float(voxel_spacing[0]),
        "spacing_y": float(voxel_spacing[1]),
        "spacing_z": float(voxel_spacing[2]) if len(voxel_spacing) > 2 else 1.0,
        "fov_x_mm": data.shape[0] * float(voxel_spacing[0]),
        "fov_y_mm": data.shape[1] * float(voxel_spacing[1]),
        "fov_z_mm": data.shape[2] * float(voxel_spacing[2]) if len(data.shape) > 2 else 0,
        "orientation": "".join(nib.aff2axcodes(img_nii.affine)),
        "dtype": str(header.get_data_dtype()),
        "intensity_min": float(np.min(data)),
        "intensity_max": float(np.max(data)),
        "intensity_mean": float(np.mean(data)),
        "intensity_std": float(np.std(data)),
    }

    # Bitdepth info
    try:
        stats["bitpix"] = int(header["bitpix"])
    except Exception:
        stats["bitpix"] = 0

    # TR, TE from header (may not be present in all NIfTI files)
    try:
        pixdim = header["pixdim"]
        if len(pixdim) > 4 and pixdim[4] > 0:
            stats["tr_sec"] = float(pixdim[4])
        else:
            stats["tr_sec"] = None
    except Exception:
        stats["tr_sec"] = None

    if label_path and os.path.exists(label_path):
        label_nii = nib.load(label_path)
        label_data = np.round(label_nii.get_fdata()).astype(np.uint8)

        unique_labels = np.unique(label_data)
        stats["unique_labels"] = str(list(unique_labels))

        total_voxels = label_data.size

        # Background
        bg_voxels = (label_data == 0).sum()
        stats["bg_percent"] = round(100 * bg_voxels / total_voxels, 2)

        # Liver (class 1 + class 2, since tumor is inside liver)
        liver_voxels = (label_data >= 1).sum()
        stats["liver_voxels"] = int(liver_voxels)
        stats["liver_percent"] = round(100 * liver_voxels / total_voxels, 2)
        stats["liver_volume_ml"] = round(
            liver_voxels * np.prod(voxel_spacing) / 1000, 2
        )

        # Tumor (class 2)
        tumor_voxels = (label_data == 2).sum()
        stats["tumor_voxels"] = int(tumor_voxels)
        stats["tumor_percent"] = round(100 * tumor_voxels / total_voxels, 4)
        stats["tumor_volume_ml"] = round(
            tumor_voxels * np.prod(voxel_spacing) / 1000, 2
        )

        # Lesion analysis
        from skimage.measure import label as sk_label, regionprops

        tumor_mask = (label_data == 2)
        labeled = sk_label(tumor_mask)
        props = regionprops(labeled)
        stats["n_lesions"] = len(props)

        lesion_diameters = []
        lesion_volumes = []
        for p in props:
            # Physical diameter from bounding box
            bbox = p.bbox  # (min_row, min_col, min_slice, max_row, max_col, max_slice)
            extents = np.array([
                (bbox[3] - bbox[0]) * voxel_spacing[0],
                (bbox[4] - bbox[1]) * voxel_spacing[1],
                (bbox[5] - bbox[2]) * voxel_spacing[2],
            ])
            diameter_mm = np.sqrt(np.sum(extents ** 2))
            diameter_cm = diameter_mm / 10.0
            lesion_diameters.append(diameter_cm)

            vol_ml = p.area * np.prod(voxel_spacing) / 1000
            lesion_volumes.append(vol_ml)

        stats["lesion_diameters_cm"] = lesion_diameters
        stats["lesion_volumes_ml"] = lesion_volumes
    else:
        stats["unique_labels"] = None
        stats["n_lesions"] = None
        stats["lesion_diameters_cm"] = []
        stats["lesion_volumes_ml"] = []

    return stats


def format_range(values, unit="", decimals=2):
    if not values:
        return "N/A"
    fmt = f".{decimals}f"
    return f"{min(values):{fmt}} - {max(values):{fmt}}{unit} (mean: {np.mean(values):{fmt}}{unit}, std: {np.std(values):{fmt}}{unit})"


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python dataset_statistics.py <image_dir>")
        print("  python dataset_statistics.py <image_dir> <label_dir>")
        print("  python dataset_statistics.py <image_dir> <label_dir> <mapping_csv>")
        sys.exit(1)

    image_dir = sys.argv[1]
    label_dir = sys.argv[2] if len(sys.argv) >= 3 else None
    mapping_csv = sys.argv[3] if len(sys.argv) >= 4 else None

    if not os.path.exists(image_dir):
        print(f"ERROR: Image directory not found: {image_dir}")
        sys.exit(1)

    # Load mapping if provided
    mapping = {}
    if mapping_csv and os.path.exists(mapping_csv):
        with open(mapping_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mapping[row["nnunet_name"]] = row["original_name"]

    img_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".nii.gz")])

    if not img_files:
        print("No .nii.gz files found")
        sys.exit(1)

    print(f"Analyzing {len(img_files)} volumes...")

    all_stats = []
    all_lesion_diameters = []
    all_lesion_volumes = []

    for i, img_file in enumerate(img_files):
        img_path = os.path.join(image_dir, img_file)

        # Find corresponding label
        label_path = None
        if label_dir:
            if mapping:
                # Use mapping: image might be case_001_0000.nii.gz -> original_name
                orig_name = mapping.get(img_file)
                if orig_name:
                    label_path = os.path.join(label_dir, orig_name)
            else:
                # Try same name, or without _0000 suffix
                label_candidate = os.path.join(label_dir, img_file)
                label_candidate2 = os.path.join(
                    label_dir, img_file.replace("_0000.nii.gz", ".nii.gz")
                )
                if os.path.exists(label_candidate):
                    label_path = label_candidate
                elif os.path.exists(label_candidate2):
                    label_path = label_candidate2

            if label_path and not os.path.exists(label_path):
                label_path = None

        stats = analyze_volume(img_path, label_path)
        all_stats.append(stats)
        all_lesion_diameters.extend(stats["lesion_diameters_cm"])
        all_lesion_volumes.extend(stats["lesion_volumes_ml"])

        print(f"  [{i+1}/{len(img_files)}] {img_file} - shape: {stats['shape_x']}x{stats['shape_y']}x{stats['shape_z']}, "
              f"spacing: {stats['spacing_x']:.2f}x{stats['spacing_y']:.2f}x{stats['spacing_z']:.2f}mm"
              + (f", lesions: {stats['n_lesions']}" if stats['n_lesions'] is not None else ""))

    # ============================================================
    # Write per-volume CSV
    # ============================================================
    csv_path = os.path.join(image_dir, "dataset_statistics.csv")
    csv_cols = [
        "filename", "shape_x", "shape_y", "shape_z",
        "spacing_x", "spacing_y", "spacing_z",
        "fov_x_mm", "fov_y_mm", "fov_z_mm",
        "orientation", "dtype", "bitpix",
        "intensity_min", "intensity_max", "intensity_mean", "intensity_std",
    ]
    if label_dir:
        csv_cols.extend([
            "unique_labels", "bg_percent", "liver_voxels", "liver_percent",
            "liver_volume_ml", "tumor_voxels", "tumor_percent", "tumor_volume_ml",
            "n_lesions",
        ])

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols, delimiter="|", extrasaction="ignore")
        writer.writeheader()
        for s in all_stats:
            writer.writerow(s)

    # ============================================================
    # Generate summary text
    # ============================================================
    shapes_x = [s["shape_x"] for s in all_stats]
    shapes_y = [s["shape_y"] for s in all_stats]
    shapes_z = [s["shape_z"] for s in all_stats]
    sp_x = [s["spacing_x"] for s in all_stats]
    sp_y = [s["spacing_y"] for s in all_stats]
    sp_z = [s["spacing_z"] for s in all_stats]
    fov_x = [s["fov_x_mm"] for s in all_stats]
    fov_y = [s["fov_y_mm"] for s in all_stats]
    fov_z = [s["fov_z_mm"] for s in all_stats]
    orientations = set(s["orientation"] for s in all_stats)
    dtypes = set(s["dtype"] for s in all_stats)
    int_mins = [s["intensity_min"] for s in all_stats]
    int_maxs = [s["intensity_max"] for s in all_stats]

    summary = []
    summary.append("=" * 70)
    summary.append("DATASET STATISTICS SUMMARY")
    summary.append("=" * 70)
    summary.append("")
    summary.append(f"Number of volumes: {len(all_stats)}")
    summary.append("")

    summary.append("--- ACQUISITION PARAMETERS ---")
    summary.append(f"Matrix size (x):        {format_range(shapes_x, '', 0)}")
    summary.append(f"Matrix size (y):        {format_range(shapes_y, '', 0)}")
    summary.append(f"Number of slices (z):   {format_range(shapes_z, '', 0)}")
    summary.append(f"In-plane resolution x:  {format_range(sp_x, 'mm')}")
    summary.append(f"In-plane resolution y:  {format_range(sp_y, 'mm')}")
    summary.append(f"Slice thickness:        {format_range(sp_z, 'mm')}")
    summary.append(f"FOV x:                  {format_range(fov_x, 'mm', 1)}")
    summary.append(f"FOV y:                  {format_range(fov_y, 'mm', 1)}")
    summary.append(f"FOV z:                  {format_range(fov_z, 'mm', 1)}")
    summary.append(f"Orientations:           {', '.join(orientations)}")
    summary.append(f"Data types:             {', '.join(dtypes)}")
    summary.append(f"Intensity range:        {min(int_mins):.1f} to {max(int_maxs):.1f}")
    summary.append("")

    if label_dir:
        liver_vols = [s["liver_volume_ml"] for s in all_stats if s.get("liver_volume_ml") is not None]
        tumor_vols_total = [s["tumor_volume_ml"] for s in all_stats if s.get("tumor_volume_ml") is not None]
        liver_pcts = [s["liver_percent"] for s in all_stats if s.get("liver_percent") is not None]
        tumor_pcts = [s["tumor_percent"] for s in all_stats if s.get("tumor_percent") is not None]
        n_lesions_list = [s["n_lesions"] for s in all_stats if s["n_lesions"] is not None]

        summary.append("--- SEGMENTATION STATISTICS ---")
        summary.append(f"Liver volume:           {format_range(liver_vols, 'ml')}")
        summary.append(f"Liver % of volume:      {format_range(liver_pcts, '%')}")
        summary.append(f"Total tumor volume:     {format_range(tumor_vols_total, 'ml')}")
        summary.append(f"Tumor % of volume:      {format_range(tumor_pcts, '%', 4)}")
        summary.append("")

        summary.append("--- LESION ANALYSIS ---")
        summary.append(f"Total lesions:          {len(all_lesion_diameters)}")
        summary.append(f"Lesions per patient:    {format_range(n_lesions_list, '', 1)}")

        if all_lesion_diameters:
            summary.append(f"Lesion diameter:        {format_range(all_lesion_diameters, 'cm')}")
            summary.append(f"Lesion volume:          {format_range(all_lesion_volumes, 'ml')}")
            summary.append("")

            # Size group distribution
            size_groups = defaultdict(int)
            for d in all_lesion_diameters:
                size_groups[get_size_group(d)] += 1

            summary.append("Lesion size distribution:")
            for g in ["<2cm", "2-3cm", "3-4cm", "4-8cm"]:
                n = size_groups.get(g, 0)
                pct = 100 * n / len(all_lesion_diameters) if all_lesion_diameters else 0
                summary.append(f"  {g:>6}: {n:>4} ({pct:.1f}%)")

        summary.append("")

    # Paper-ready paragraph
    summary.append("--- PAPER-READY DESCRIPTION ---")
    summary.append("")
    sp_x_range = f"{min(sp_x):.2f}-{max(sp_x):.2f}" if min(sp_x) != max(sp_x) else f"{min(sp_x):.2f}"
    sp_y_range = f"{min(sp_y):.2f}-{max(sp_y):.2f}" if min(sp_y) != max(sp_y) else f"{min(sp_y):.2f}"
    sp_z_range = f"{min(sp_z):.2f}-{max(sp_z):.2f}" if min(sp_z) != max(sp_z) else f"{min(sp_z):.2f}"

    paragraph = (
        f"The dataset consists of {len(all_stats)} MRI volumes "
        f"with matrix sizes of {min(shapes_x)}-{max(shapes_x)} x "
        f"{min(shapes_y)}-{max(shapes_y)} x {min(shapes_z)}-{max(shapes_z)} voxels. "
        f"The in-plane resolution ranged from {sp_x_range} x {sp_y_range} mm "
        f"with slice thickness of {sp_z_range} mm."
    )

    if label_dir and all_lesion_diameters:
        size_groups = defaultdict(int)
        for d in all_lesion_diameters:
            size_groups[get_size_group(d)] += 1

        paragraph += (
            f" A total of {len(all_lesion_diameters)} lesions were identified "
            f"across all volumes (mean: {np.mean(n_lesions_list):.1f} +/- {np.std(n_lesions_list):.1f} per patient). "
            f"The lesion diameters ranged from {min(all_lesion_diameters):.2f} to "
            f"{max(all_lesion_diameters):.2f} cm "
            f"(mean: {np.mean(all_lesion_diameters):.2f} +/- {np.std(all_lesion_diameters):.2f} cm), "
            f"with {size_groups.get('<2cm', 0)} ({100*size_groups.get('<2cm', 0)/len(all_lesion_diameters):.1f}%) "
            f"smaller than 2 cm, "
            f"{size_groups.get('2-3cm', 0)} ({100*size_groups.get('2-3cm', 0)/len(all_lesion_diameters):.1f}%) "
            f"between 2-3 cm, "
            f"{size_groups.get('3-4cm', 0)} ({100*size_groups.get('3-4cm', 0)/len(all_lesion_diameters):.1f}%) "
            f"between 3-4 cm, and "
            f"{size_groups.get('4-8cm', 0)} ({100*size_groups.get('4-8cm', 0)/len(all_lesion_diameters):.1f}%) "
            f"between 4-8 cm."
        )

    summary.append(paragraph)
    summary.append("")

    # Print and save
    summary_text = "\n".join(summary)
    print(summary_text)

    summary_path = os.path.join(image_dir, "dataset_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)

    print(f"\nPer-volume stats saved to: {csv_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
