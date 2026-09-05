"""
Postprocessing for nnU-Net predictions:
  - Label 1 (liver): keep only the largest connected component
  - Label 2 (tumor): keep only components that are inside the liver mask

Usage:
  python postprocess_predictions.py -i /path/to/predictions -o /path/to/output

  If -o is not provided, results are saved in <input_folder>_postprocessed
"""

import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.ndimage import label as scipy_label


def keep_largest_component(binary_mask):
    """Keep only the largest connected component in a binary mask."""
    labeled_array, num_features = scipy_label(binary_mask)
    if num_features == 0:
        return binary_mask

    component_sizes = np.bincount(labeled_array.ravel())
    # Ignore background (index 0)
    component_sizes[0] = 0
    largest_label = component_sizes.argmax()
    return (labeled_array == largest_label).astype(binary_mask.dtype)


def postprocess_segmentation(seg_data):
    """
    Postprocess a segmentation volume:
      - Label 1 (liver): keep largest connected component
      - Label 2 (tumor): remove tumors outside the liver
    """
    result = np.zeros_like(seg_data)

    # Step 1: Keep largest connected component for liver (label >= 1, since tumor is inside liver)
    liver_mask = (seg_data >= 1).astype(np.uint8)
    liver_cleaned = keep_largest_component(liver_mask)

    # Step 2: Assign liver label
    result[liver_cleaned == 1] = 1

    # Step 3: Restore tumor labels only where they overlap with cleaned liver
    tumor_mask = (seg_data == 2).astype(np.uint8)
    tumor_in_liver = tumor_mask * liver_cleaned
    result[tumor_in_liver == 1] = 2

    return result


def process_file(input_path, output_path):
    """Process a single NIfTI file."""
    img = nib.load(str(input_path))
    seg_data = img.get_fdata().astype(np.uint8)

    original_labels = np.unique(seg_data)
    seg_postprocessed = postprocess_segmentation(seg_data)
    new_labels = np.unique(seg_postprocessed)

    # Report changes
    original_liver = np.sum(seg_data >= 1)
    new_liver = np.sum(seg_postprocessed >= 1)
    removed_voxels = original_liver - new_liver
    if removed_voxels > 0:
        print(f"  Removed {removed_voxels} voxels ({removed_voxels/original_liver*100:.1f}%) from liver")

    original_tumor = np.sum(seg_data == 2)
    new_tumor = np.sum(seg_postprocessed == 2)
    removed_tumor = original_tumor - new_tumor
    if removed_tumor > 0:
        print(f"  Removed {removed_tumor} tumor voxels outside liver")

    # Save with same header/affine
    out_img = nib.Nifti1Image(seg_postprocessed, img.affine, img.header)
    nib.save(out_img, str(output_path))


def main():
    parser = argparse.ArgumentParser(description='Postprocess nnU-Net predictions: keep largest liver component')
    parser.add_argument('-i', '--input', required=True, help='Input folder with predicted .nii.gz files')
    parser.add_argument('-o', '--output', default=None, help='Output folder (default: <input>_postprocessed)')
    args = parser.parse_args()

    input_dir = Path(args.input)
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = input_dir.parent / (input_dir.name + '_postprocessed')

    output_dir.mkdir(parents=True, exist_ok=True)

    nifti_files = sorted(list(input_dir.glob('*.nii.gz')) + list(input_dir.glob('*.nii')))
    print(f"Found {len(nifti_files)} files to process")

    for f in nifti_files:
        print(f"Processing: {f.name}")
        process_file(f, output_dir / f.name)

    print(f"\nDone. Postprocessed files saved to: {output_dir}")


if __name__ == '__main__':
    main()
