import nibabel as nib
import numpy as np
import os
import sys

folder = sys.argv[1]

if not os.path.exists(folder):
    print(f"ERROR: Folder does not exist: {folder}")
    sys.exit(1)

files = sorted([f for f in os.listdir(folder) if f.endswith(".nii.gz")])
print(f"Found {len(files)} .nii.gz files in {folder}")

if len(files) == 0:
    print("WARNING: No .nii.gz files found!")
    sys.exit(1)

for i, filename in enumerate(files):
    filepath = os.path.join(folder, filename)
    try:
        print(f"[{i+1}/{len(files)}] Processing: {filename}")
        img = nib.load(filepath)
        data = np.asarray(img.dataobj)
        data_rounded = np.round(data).astype(np.int16)
        print(f"  unique values before={np.unique(data)}, after={np.unique(data_rounded)}")
        img.header.set_data_dtype(np.int16)
        new_img = nib.Nifti1Image(data_rounded, img.affine, img.header)
        new_img.set_sform(img.header.get_sform(), code=img.header['sform_code'])
        new_img.set_qform(img.header.get_qform(), code=img.header['qform_code'])
        nib.save(new_img, filepath)
        print(f"  Saved: {filepath}")
    except Exception as e:
        print(f"  ERROR processing {filename}: {e}")

print("Done!")
