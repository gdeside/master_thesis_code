from dipy.io.image import load_nifti, save_nifti
import matplotlib.pyplot as plt
from pathlib import Path
import os
import numpy as np
from dipy.io.gradients import read_bvals_bvecs
from scipy import stats
import pandas as pd
import seaborn as sns
import numpy.ma as ma
import pingouin as pg 
import matplotlib.gridspec as gridspec

import cv2 as cv
from typing import Tuple

import csv

def visualize_patient_T1_MRI(patient, show=False, save_as_png=False):
    """
    Visualize T1 MRI data for a specific patient in sagittal and axial views.

    Parameters:
    - patient (str): The name of the patient.
    - show (bool): Whether to display the plot using plt.show(). Default is True.
    - save_as_png (bool): Whether to save the plot as a PNG file. Default is False.

    Returns:
    - None
    """

    folder_path = os.path.abspath(os.path.dirname(os.getcwd()))

    print(f"Starting T1 MRI data visualization for patient: {patient}")
    
    subject_folder = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient}"
    T1_subject = f'{subject_folder}/T1/{patient}_T1_brain.nii.gz'
    print(f"T1 MRI image path set to: {T1_subject}")

    data, _, _ = load_nifti(T1_subject, return_img=True)
    print("Data loaded successfully.")

    print("Data Shape:", data.shape)

    axial_middle = data.shape[2] // 2
    print(f"Preparing plot for patient: {patient}")

    plt.figure(f"{patient} T1 MRI Visualization")
    plt.suptitle(f'Visualization for {patient} T1 MRI')  # Title for the entire plot

    plt.subplot(1, 2, 1).set_axis_off()
    plt.imshow(data[:, :, axial_middle], cmap='gray', origin='lower')
    plt.title('Sagittal View')  # Subtitle for the sagittal view subplot

    plt.subplot(1, 2, 2).set_axis_off()
    plt.imshow(data[:, axial_middle, :], cmap='gray', origin='lower')
    plt.title('Axial View')  # Subtitle for the axial view subplot

    if show:
        print("Displaying the plot.")
        plt.show()

    if save_as_png:
        file_path = f'{folder_path}/images/T1/{patient}_T1_MRI.png'
        print(f"Preparing to save the plot as PNG at: {file_path}")
        if not os.path.exists(Path(file_path).parent):
            os.makedirs(Path(file_path).parent)
            print(f"Created directories for storing the plot: {Path(file_path).parent}")
        plt.savefig(file_path, bbox_inches='tight')
        print("Saved T1 MRI Visualization:", file_path)

    plt.close()
    print("Finished T1 MRI data visualization.")
    return 


def visualize_T1_and_bvalue_differences(patient_path, prepoc_raw="preproc", show=False, save_as_png=False):
    """
    Visualize differences in T1 and diffusion-weighted images (DWIs) for various b-values 
    for a specific patient. This function plots a T1 image alongside DWI slices for selected 
    b-values to highlight differences.

    Parameters:
    - patient_path (str): The name of the patient.
    - prepoc_raw (str): Specify whether to use preprocessed ("preproc") or raw ("raw") data.
    - show (bool): Whether to display the plot. Default is False.
    - save_as_png (bool): Whether to save the plot as a PNG file. Default is False.

    Returns:
    - None (plots the results)
    """

    print(f"Starting visualization of T1 and DWI b-value differences for patient: {patient_path}, prepoc/raw: {prepoc_raw}")

    # Setup file paths and load data
    folder_path = os.path.abspath(os.path.dirname(os.getcwd()))
    subject_folder = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}"
    T1_subject = f'{subject_folder}/reg/{patient_path}_T1_DWIspace.nii.gz'

    print(f"T1 image path: {T1_subject}")
    data_T1, affine_T1 = load_nifti(T1_subject)

    # Plot setup
    b_values_lst = [0, 1000, 3000, 5000, 10000]
    fig, axes = plt.subplots(2, (len(b_values_lst) + 1)//2, figsize=(10,10))

    axes = axes.flatten()
    # Plot T1
    data_T1_slice = data_T1[:, :, data_T1.shape[2] // 2]  # Use a middle slice
    axes[0].imshow(data_T1_slice.T, cmap="gray", origin='lower')
    axes[0].set_title('T1 Image')
    axes[0].axis('off')

    # Load DWI data
    fdwi = f"{subject_folder}/dMRI/{prepoc_raw}/{patient_path}_dmri_preproc.nii.gz"
    fbval = f"{subject_folder}/dMRI/{prepoc_raw}/{patient_path}_dmri_preproc.bval"
    fbvec = f"{subject_folder}/dMRI/{prepoc_raw}/{patient_path}_dmri_preproc.bvec"

    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    data_dwi, affine_dwi, voxel_size = load_nifti(fdwi, return_voxsize=True)

    # Plot DWI for each b-value
    for i, b_value in enumerate(b_values_lst):
        target_indices = np.where((abs(bvals - b_value) <= 150))[0]
        # Assuming we're visualizing the same middle slice for each b-value
        target_data = data_dwi[:, :, data_dwi.shape[2] // 2, target_indices[0]]  # Use the first index for simplicity
        axes[i + 1].imshow(target_data.T, cmap="gray", origin='lower')
        axes[i + 1].set_title(f'b={b_value}')
        axes[i + 1].axis('off')

    # Adjust layout and show/save plot
    plt.tight_layout()
    if show:
        plt.show()
    if save_as_png:
        save_path = os.path.join(folder_path, "images/T1/", f"T1_bvalues_diff_{patient_path}_{prepoc_raw}.png")
        plt.savefig(save_path)
        print(f"Plot saved as PNG at: {save_path}")

    plt.close()
    print("Visualization completed.")


def overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.5
) -> np.ndarray:
    """Combines a grayscale image and its segmentation mask into a single image.
    
    Params:
        image: Grayscale training image (2D array).
        mask: Segmentation mask (2D array of same shape as `image`).
        color: RGB color for segmentation mask rendering.
        alpha: Segmentation mask's transparency.
    
    Returns:
        image_combined: The combined image with overlayed mask.
    """
    # Create a 3-channel version of the grayscale image
    grayscale_rgb = np.stack([image] * 3, axis=-1).astype(float)
    
    # Convert the mask to a 3-channel mask with the specified color
    mask_rgb = np.stack([mask * color[0], mask * color[1], mask * color[2]], axis=-1).astype(float)
    
    # Blend the grayscale image with the colored mask
    image_combined = cv.addWeighted(grayscale_rgb, 1 - alpha, mask_rgb, alpha, 0).astype(np.uint8)
    
    return image_combined




def color_ROI_T1(patient_path, atlas_name, atlas_values,roi_names):
    """
    Visualizes a T1-weighted anatomical image with overlaid ROIs (Regions of Interest) colored based on their values.

    Args:
    - patient_path (str): The patient identifier.
    - atlas_name (str): Name of the atlas used for ROI segmentation.
    - atlas_values (list of int): List of atlas-specific integers representing the ROIs.
    - save_as_png (bool): If True, saves the plot as a PNG file.

    Returns:
    - None: Displays or saves the plot.
    """
    print(f"Starting visualization of T1 and DWI b-value differences for patient: {patient_path}")

    # Setup file paths and load data
    folder_path = os.path.abspath(os.path.dirname(os.getcwd()))
    subject_folder = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}"
 


    atlas_path = f'{subject_folder}/reg/{patient_path}_Atlas_{atlas_name}_InSubjectT1Space.nii.gz'
    atlas, affine = load_nifti(atlas_path)
    print("Atlas loaded successfully.")


    def is_in_range(atlas, atlas_values, tolerance=0.1):
        # Create a boolean array initialized to False
        in_range = np.zeros_like(atlas, dtype=bool)

        # Iterate over each value in atlas_values
        for val in atlas_values:
            # Create a mask that identifies elements in the range [val - tolerance, val + tolerance]
            range_mask = (atlas >= (val - tolerance)) & (atlas <= (val + tolerance))
            # Combine this mask with the existing boolean array
            in_range = in_range | range_mask

        return in_range

    tolerance = 0.1  # Set the desired tolerance

    # Create a reduced atlas array with the tolerance
    atlas_reduced = np.where(is_in_range(atlas, atlas_values, tolerance), atlas, 0)

    print(np.unique(atlas_reduced, return_counts=True))


    atlas_reduced_path = f'{subject_folder}/reg/{patient_path}_Atlas_{atlas_name}_InSubjectT1Space_{roi_names}.nii.gz'

    save_nifti(atlas_reduced_path, atlas_reduced.astype(np.float32), affine)
    print("Saved:", atlas_reduced_path)








############################################################################################################################################################################


tract_dictionary = {
    1: "unclassified",
    2: "middle cerebellar peduncle",
    3: "pontine crossing tract",
    4: "genu of corpus callosum",
    5: "body of corpus callosum",
    6: "splenium of corpus callosum",
    7: "fornix",
    8: "corticospinal tract r",
    9: "corticospinal tract l",
    10: "medial lemniscus r",
    11: "medial lemniscus l",
    12: "inferior cerebellar peduncle r",
    13: "inferior cerebellar peduncle l",
    14: "superior cerebellar peduncle r",
    15: "superior cerebellar peduncle l",
    16: "cerebral peduncle r",
    17: "cerebral peduncle l",
    18: "anterior limb of internal capsule r",
    19: "anterior limb of internal capsule l",
    20: "posterior limb of internal capsule r",
    21: "posterior limb of internal capsule l",
    22: "retrolenticular part of internal capsule r",
    23: "retrolenticular part of internal capsule l",
    24: "anterior corona radiata r",
    25: "anterior corona radiata l",
    26: "superior corona radiata r",
    27: "superior corona radiata l",
    28: "posterior corona radiata r",
    29: "posterior corona radiata l",
    30: "posterior thalamic radiation", # appears twice
    31: "sagittal stratum", # appears twice
    32: "external capsule r",
    33: "external capsule l",
    34: "cingulum", # cingulum appears four times (36 to 39)
    35: "cingulum",
    36: "cingulum",
    37: "cingulum",
    38: "fornix", # fornix appears twice (7 and 41)
    39: "fornix",
    40: "superior longitudinal fasciculus r",
    41: "superior longitudinal fasciculus l",
    42: "superior fronto-occipital fasciculus", # appears twice (44 and 45)
    43: "superior fronto-occipital fasciculus",
    44: "uncinate fasciculus r",
    45: "uncinate fasciculus l",
    46: "tapetum r",
    47: "tapetum l"
}


atlas_name = "JHU-ICBM-labels-1mm"

atlas_values = [5]

roi_names = "body_of_corpus_callosum"



for i in range(1001,1002):
    patient_path = f"sub-{i}"

    #visualize_patient_T1_MRI(patient_path, show=False, save_as_png=True)

    #visualize_T1_and_bvalue_differences(patient_path, prepoc_raw="preproc", show=False, save_as_png=True)

    
    color_ROI_T1(patient_path, atlas_name, [5,20,21]," body of corpus callosum posterior limb of internal capsule")