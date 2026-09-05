"""
1. Loads nifti images and corresponding ground-truth segmentations.
2. Creates a spherical structuring element (diameter=2).
3. Dilation is applied to voxels labeled as 2.
4. Saves updated volumes to a specified output folder.
"""


import nibabel as nib
import matplotlib.pyplot as plt 
import numpy as np
import os
import scipy
import scipy.misc
from scipy import ndimage
from scipy.signal import find_peaks
from scipy.ndimage import binary_dilation
from pathlib import Path

def ball_structure(diameter=2):
    """
    Creates a 3D spherical structuring element with the specified diameter.
    1. diameter=2 → radius=1
    2. Build a (2*radius+1)^3 grid
    3. Mark the voxels whose distance from center ≤ radius
    """
    r = diameter // 2
    size = 2 * r + 1
    center = r
    x, y, z = np.ogrid[:size, :size, :size]
    dist_sq = (x - center)**2 + (y - center)**2 + (z - center)**2
    return dist_sq <= r*r

# Define input and output directories.
# input_folder_img = 'D:\\Yuliia_data\\Yuliia_data\\COMET_2023\\MR_img_and_seg_initial_batch\\img\\all'
# input_folder_gt = 'D:\\Yuliia_data\\Yuliia_data\\COMET_2023\\MR_img_and_seg_initial_batch\\seg_revised_2023\\all'
# output_folder = Path('D:\\Yuliia_data\\Yuliia_data\\COMET_2023\\MR_img_and_seg_initial_batch\\seg_revised_2023\\tumors_dilated_3')
# output_folder.mkdir(parents=True, exist_ok=True)

input_folder_img = 'D:\\Yuliia_data\\Yuliia_data\\COMET_2023\\MR_img_seg_last_96volumes\\img\\'
input_folder_gt = 'D:\\Yuliia_data\\Yuliia_data\\COMET_2023\\MR_img_seg_last_96volumes\\seg_2class'
output_folder = Path('D:\\Yuliia_data\\Yuliia_data\\COMET_2023\\MR_img_seg_last_96volumes\\tumors_dilated_3')
output_folder.mkdir(parents=True, exist_ok=True)

# Create the spherical structuring element (radius=1 → diameter=2).
struct = ball_structure(3)

# List all filenames in the input_folder_img.
x_filenames = os.listdir(input_folder_img)

# Iterate over each file by name.
for name in x_filenames:
    # 1. Load the image volume.
    dir_path = os.path.join(input_folder_img, name)
    img_orig = nib.load(dir_path)
    img_matrix = img_orig.get_fdata()

    # 2. Load the corresponding ground truth volume.
    dir_path_gt = os.path.join(input_folder_gt, name)
    try: 
        gt_orig = nib.load(dir_path_gt)
        gt_matrix = gt_orig.get_fdata()  

        # 3. Copy ground truth data to a new matrix for in-place modification.
        mymatrix_new = gt_matrix

        # 4. Create a binary mask where label=2.
        mask = (gt_matrix == 2)

        # 5. Dilate the mask using the spherical structuring element.
        dilated = binary_dilation(mask, structure=struct)

        # 6. Assign label=2 to all dilated voxels.
        mymatrix_new[dilated] = 2

        # 7. Save updated volume as a new nifti file in the output folder.
        header = nib.Nifti1Header()
        image = nib.Nifti1Image(mymatrix_new, img_orig.affine, header)
        save_path = os.path.join(output_folder, name)
        print('saving new segmentation', save_path)
        nib.save(image, save_path)
    except FileNotFoundError:
        print(f"Ground truth not found: {dir_path_gt}, skipping.")
        continue

