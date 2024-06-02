"""
File: patient_dti_analysis.py
Author: Guillaume Francois Deside

Description:
This file contains functions for processing and analyzing Diffusion Tensor Imaging (DTI) data.
"""


import numpy as np
import datetime
import os
from dipy.io.image import load_nifti, save_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti
from dipy.segment.mask import median_otsu
import matplotlib.pyplot as plt

def process_dti(patient_path, b0_threshold=60, bet_median_radius=2, bet_numpass=1, bet_dilate=2, prepoc_raw="preproc"):
    """
    Process Diffusion Tensor Imaging (DTI) data for an individual patient.

    This function performs several steps to process the DTI data of a given patient, including:
    1. Loading the DWI data, b-values, and b-vectors.
    2. Filtering the DWI data to select specific b-values.
    3. Applying brain extraction and generating a brain mask.
    4. Fitting a tensor model to the DWI data.
    5. Calculating and saving DTI metrics (FA, MD, RD, AD, eigenvectors, eigenvalues, RGB FA, and residuals).

    Parameters:
    - patient_path (str): The patient identifier.
    - b0_threshold (int, optional): B0 threshold for gradient table estimation. Default is 60.
    - bet_median_radius (int, optional): Median radius for brain extraction. Default is 2.
    - bet_numpass (int, optional): Number of passes for brain extraction. Default is 1.
    - bet_dilate (int, optional): Dilation factor for brain mask. Default is 2.
    - prepoc_raw (str, optional): Specify whether to use preprocessed ("preproc") or raw ("raw") data. Default is "preproc".

    Returns:
    - None
    """
    log_prefix = "DTI SOLO"
    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + f": Beginning of individual DTI processing for patient {patient_path} \n")

    folder_path = os.path.dirname(os.getcwd())
    dti_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/DTI/"
    images_path = folder_path + "/images/"

    if not os.path.exists(dti_path):
        os.makedirs(dti_path)
    
    print("Folder Path:", folder_path)
    print("DTI Path:", dti_path)
    print("Images Path:", images_path)
    print("Preprocessing Type:", prepoc_raw)

    fdwi = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/{prepoc_raw}/{patient_path}_dmri_preproc.nii.gz"
    fbval = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/{prepoc_raw}/{patient_path}_dmri_preproc.bval"
    fbvec = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/{prepoc_raw}/{patient_path}_dmri_preproc.bvec"

    print("DWI File Path:", fdwi)
    print("BVAL File Path:", fbval)
    print("BVEC File Path:", fbvec)

    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    data, affine, voxel_size = load_nifti(fdwi, return_voxsize=True)


    target_indices = np.where((abs(bvals - 1000) <= 150) | (bvals == 0) | (abs(bvals - 3000) <= 150) )[0]


    # Select data and bvecs for the target b-value
    data = data[:, :, :, target_indices]
    bvals = bvals[target_indices]
    bvecs = bvecs[target_indices, :]

    print("Data shape ",np.shape(data))

    b0_mask, mask = median_otsu(data, median_radius=bet_median_radius, numpass=bet_numpass,
                                vol_idx=range(0, np.shape(data)[3]), dilate=bet_dilate)

    b0_threshold = np.min(bvals) + 10
    b0_threshold = max(50, b0_threshold)
    gtab = gradient_table(bvals, bvecs, b0_threshold=b0_threshold)

    tenmodel = dti.TensorModel(gtab)
    tenfit = tenmodel.fit(data, mask=mask)

    FA = dti.fractional_anisotropy(tenfit.evals)
    FA[np.isnan(FA)] = 0
    FA = np.clip(FA, 0, 1)
    fa_file_path = dti_path + patient_path + "_FA.nii.gz"
    save_nifti(fa_file_path, FA.astype(np.float32), affine)
    print("Saved:", fa_file_path)

    RGB = dti.color_fa(FA, tenfit.evecs)
    rgb_file_path = dti_path + patient_path + "_fargb.nii.gz"
    save_nifti(rgb_file_path, np.array(255 * RGB, 'uint8'), affine)
    print("Saved:", rgb_file_path)

    MD = dti.mean_diffusivity(tenfit.evals)
    md_file_path = dti_path + patient_path + "_MD.nii.gz"
    save_nifti(md_file_path, MD.astype(np.float32), affine)
    print("Saved:", md_file_path)

    RD = dti.radial_diffusivity(tenfit.evals)
    rd_file_path = dti_path + patient_path + "_RD.nii.gz"
    save_nifti(rd_file_path, RD.astype(np.float32), affine)
    print("Saved:", rd_file_path)

    AD = dti.axial_diffusivity(tenfit.evals)
    ad_file_path = dti_path + patient_path + "_AD.nii.gz"
    save_nifti(ad_file_path, AD.astype(np.float32), affine)
    print("Saved:", ad_file_path)

    evecs_file_path = dti_path + patient_path + "_evecs.nii.gz"
    save_nifti(evecs_file_path, tenfit.evecs.astype(np.float32), affine)
    print("Saved:", evecs_file_path)

    evals_file_path = dti_path + patient_path + "_evals.nii.gz"
    save_nifti(evals_file_path, tenfit.evals.astype(np.float32), affine)
    print("Saved:", evals_file_path)

    dtensor_file_path = dti_path + patient_path + "_dtensor.nii.gz"
    save_nifti(dtensor_file_path, tenfit.quadratic_form.astype(np.float32), affine)
    print("Saved:", dtensor_file_path)

    reconstructed = tenfit.predict(gtab, S0=data[..., 0])
    residual = data - reconstructed
    residual_file_path = dti_path + patient_path + "_residual.nii.gz"
    save_nifti(residual_file_path, residual.astype(np.float32), affine)
    print("Saved:", residual_file_path)

    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + f": DTI processing completed for patient {patient_path}.\n")
    return 


def calculate_and_plot_dti_mse(patient_path, prepoc_raw="preproc", save_as_png=False, show=False):
    """
    Calculate and plot Mean Squared Error (MSE) for Diffusion Tensor Imaging (DTI).

    This function performs the following steps:
    1. Sets up file paths and directories.
    2. Loads the residuals from the DTI analysis.
    3. Calculates the MSE from the residuals.
    4. Prints statistical information about the MSE.
    5. Saves the MSE as a NIfTI file.
    6. Plots the MSE and optionally saves the plot as a PNG file.

    Parameters:
    - patient_path (str): The name of the patient.
    - prepoc_raw (str, optional): Specify whether to use preprocessed ("preproc") or raw ("raw") data. Default is "preproc".
    - save_as_png (bool, optional): Whether to save the plot as a PNG file. Default is False.
    - show (bool, optional): Whether to display the plot using plt.show(). Default is False.

    Returns:
    - None
    """
    folder_path = os.path.dirname(os.getcwd())
    dti_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/DTI/"
    images_path = folder_path + "/images/"

    print("Folder Path:", folder_path)
    print("DTI Path:", dti_path)
    print("Images Path:", images_path)
    print("Preprocessing Type:", prepoc_raw)

    residual_path = dti_path + patient_path + "_residual.nii.gz"
    print("Residual Path:", residual_path)

    residual, affine = load_nifti(residual_path)

    sli = residual.shape[2] // 2

    mse = np.nanmean(residual ** 2, axis=-1)

    print(f"Statistical Information about MSE (DTI - {patient_path} - {prepoc_raw}):")
    print(f" - Mean: {np.nanmean(mse):.4f}")
    print(f" - Median: {np.nanmedian(mse):.4f}")
    print(f" - Standard Deviation: {np.nanstd(mse):.4f}")


    mse_file_path = dti_path + patient_path + "_mse.nii.gz"
    save_nifti(mse_file_path, mse.astype(np.float32), affine)
    print("Saved MSE File:", mse_file_path)

    # Plot MSE
    plt.figure(f'MSE DTI - {patient_path} - {prepoc_raw}')
    plt.imshow(mse[:, :, sli])
    plt.title(f'MSE DTI - {patient_path} - {prepoc_raw}')

    save_plot_path = f"{images_path}{patient_path}_{prepoc_raw}_MSE_dti.png"

    if save_as_png:
        plt.savefig(save_plot_path)
        print("Saved MSE Plot:", save_plot_path)

    if show:
        plt.show()

    return 



def process_dti_b_values(patient_path, b0_threshold=60, bet_median_radius=2, bet_numpass=1, bet_dilate=2, prepoc_raw="preproc"):
    """
    Process Diffusion Tensor Imaging (DTI) for an individual patient for multiple b-values.

    This function performs the following steps for each specified b-value:
    1. Sets up file paths and directories.
    2. Loads the diffusion-weighted imaging (DWI) data, b-values, and b-vectors.
    3. Selects the data corresponding to the target b-value.
    4. Performs brain extraction using median Otsu.
    5. Constructs a gradient table.
    6. Fits a tensor model to the data.
    7. Calculates and saves the DTI metrics (FA, MD, RD, AD) and their color-coded representations.
    8. Saves the eigenvalues and eigenvectors of the tensor model.
    9. Predicts the diffusion signal and calculates the residuals.
    10. Saves the residuals.

    Parameters:
    - patient_path (str): The identifier of the patient.
    - b0_threshold (int): B0 threshold for gradient table estimation.
    - bet_median_radius (int): Median radius for brain extraction.
    - bet_numpass (int): Number of passes for brain extraction.
    - bet_dilate (int): Dilation factor for brain mask.
    - prepoc_raw (str): Specify whether to use preprocessed ("preproc") or raw ("raw") data.

    Returns:
    - None
    """
    import datetime
    log_prefix = "DTI SOLO"
    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + f": Beginning of individual DTI processing for patient {patient_path} \n")

    parent_folder = os.path.dirname(os.getcwd())  # Get the parent folder of the current working directory

    dti_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/DTI/"
    images_path = os.path.join(parent_folder, "images/")

    if not os.path.exists(dti_path):
        os.makedirs(dti_path)
    
    print("Folder Path:", parent_folder)
    print("DTI Path:", dti_path)
    print("Images Path:", images_path)
    print("Preprocessing Type:", prepoc_raw)

    fdwi = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/preproc/{patient_path}_dmri_preproc.nii.gz"
    fbval = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/preproc/{patient_path}_dmri_preproc.bval"
    fbvec = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/preproc/{patient_path}_dmri_preproc.bvec"

    print("DWI File Path:", fdwi)
    print("BVAL File Path:", fbval)
    print("BVEC File Path:", fbvec)

    print("Reading b-values and b-vectors...")
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    print("Shape of b-values:", bvals.shape)
    print("Shape of b-vectors:", bvecs.shape)

    print("\nLoading DWI data...")
    data, affine, voxel_size = load_nifti(fdwi, return_voxsize=True)
    print("Shape of DWI data:", data.shape)
    print("Voxel size:", voxel_size)

    target_bvalues = [1000, 3000]

    for target_bvalue in target_bvalues:
        # Select indices corresponding to the target b-value
        target_indices = np.where((abs(bvals - target_bvalue) <= 150) | (bvals == 0))[0]

        print(f"Number of elements in target_indices: {len(target_indices)}")

        # Select data and bvecs for the target b-value
        target_data = data[:, :, :, target_indices]
        target_bvals = bvals[target_indices]
        target_bvecs = bvecs[target_indices, :]

        b0_mask, mask = median_otsu(target_data, median_radius=bet_median_radius, numpass=bet_numpass,
                                    vol_idx=range(0, np.shape(target_data)[3]), dilate=bet_dilate)
        
        b0_threshold = np.min(target_bvals) + 10
        b0_threshold = max(50, b0_threshold)
        gtab = gradient_table(target_bvals, target_bvecs, b0_threshold=b0_threshold)

        tenmodel = dti.TensorModel(gtab)
        tenfit = tenmodel.fit(target_data, mask=mask)

        FA = dti.fractional_anisotropy(tenfit.evals)
        FA[np.isnan(FA)] = 0
        FA = np.clip(FA, 0, 1)
        fa_file_path = dti_path + f"{patient_path}_FA_b{target_bvalue}.nii.gz"
        save_nifti(fa_file_path, FA.astype(np.float32), affine)
        print("Saved:", fa_file_path)

        RGB = dti.color_fa(FA, tenfit.evecs)
        rgb_file_path = dti_path + f"{patient_path}_fargb_b{target_bvalue}.nii.gz"
        save_nifti(rgb_file_path, np.array(255 * RGB, 'uint8'), affine)
        print("Saved:", rgb_file_path)

        MD = dti.mean_diffusivity(tenfit.evals)
        md_file_path = dti_path + f"{patient_path}_MD_b{target_bvalue}.nii.gz"
        save_nifti(md_file_path, MD.astype(np.float32), affine)
        print("Saved:", md_file_path)

        RD = dti.radial_diffusivity(tenfit.evals)
        rd_file_path = dti_path + f"{patient_path}_RD_b{target_bvalue}.nii.gz"
        save_nifti(rd_file_path, RD.astype(np.float32), affine)
        print("Saved:", rd_file_path)

        AD = dti.axial_diffusivity(tenfit.evals)
        ad_file_path = dti_path + f"{patient_path}_AD_b{target_bvalue}.nii.gz"
        save_nifti(ad_file_path, AD.astype(np.float32), affine)
        print("Saved:", ad_file_path)

        evecs_file_path = dti_path + f"{patient_path}_evecs_b{target_bvalue}.nii.gz"
        save_nifti(evecs_file_path, tenfit.evecs.astype(np.float32), affine)
        print("Saved:", evecs_file_path)

        evals_file_path = dti_path + f"{patient_path}_evals_b{target_bvalue}.nii.gz"
        save_nifti(evals_file_path, tenfit.evals.astype(np.float32), affine)
        print("Saved:", evals_file_path)

        dtensor_file_path = dti_path + f"{patient_path}_dtensor_b{target_bvalue}.nii.gz"
        save_nifti(dtensor_file_path, tenfit.quadratic_form.astype(np.float32), affine)
        print("Saved:", dtensor_file_path)

        reconstructed = tenfit.predict(gtab, S0=target_data[..., 0])
        residual = target_data - reconstructed
        residual_file_path = dti_path + f"{patient_path}_residual_b{target_bvalue}.nii.gz"
        save_nifti(residual_file_path, residual.astype(np.float32), affine)
        print("Saved:", residual_file_path)

    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + f": DTI processing completed for patient {patient_path}.\n")
    
    return


def calculate_and_plot_dti_mse_b_value(patient_path, prepoc_raw="preproc", save_as_png=False, show=True):
    """
    Calculate and plot Mean Squared Error (MSE) for Diffusion Tensor Imaging (DTI) at a specific b-value.

    This function performs the following steps:
    1. Sets up file paths and directories.
    2. Loads the residuals of the diffusion signal prediction.
    3. Calculates the Mean Squared Error (MSE) from the residuals.
    4. Saves the MSE as a NIfTI file.
    5. Plots the MSE and optionally saves the plot as a PNG file.

    Parameters:
    - patient_path (str): The identifier of the patient.
    - prepoc_raw (str): Specify whether to use preprocessed ("preproc") or raw ("raw") data.
    - show (bool): Whether to display the plot using plt.show(). Default is True.
    - save_as_png (bool): Whether to save the plot as a PNG file. Default is False.

    Returns:
    - None
    """
    parent_folder = os.path.dirname(os.getcwd())
    dti_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/DTI/"
    images_path = parent_folder + "/images/"

    print("Folder Path:", parent_folder)
    print("DTI Path:", dti_path)
    print("Images Path:", images_path)
    print("Preprocessing Type:", prepoc_raw)

    target_bvalues = [1000, 3000]

    for target_bvalue in target_bvalues:
        
        residual_path = dti_path + patient_path + f"_residual_b{target_bvalue}.nii.gz"
        print("Residual Path:", residual_path)
        
        residual, affine = load_nifti(residual_path)
        
        sli = residual.shape[2] // 2
        
        mse = np.nanmean(residual ** 2, axis=-1)

        print(f"Statistical Information about MSE (DTI - {patient_path} - {prepoc_raw} - B{target_bvalue}):")
        print(f" - Mean: {np.nanmean(mse):.4f}")
        print(f" - Median: {np.nanmedian(mse):.4f}")
        print(f" - Standard Deviation: {np.nanstd(mse):.4f}")

        mse_file_path = dti_path + patient_path + f"_mse_b{target_bvalue}.nii.gz"
        save_nifti(mse_file_path, mse.astype(np.float32), affine)
        print("Saved MSE File:", mse_file_path)

        # Plot MSE
        plt.figure(f'MSE DTI - {patient_path} - {prepoc_raw} - B{target_bvalue}')
        plt.imshow(mse[:, :, sli])
        plt.title(f'MSE DTI - {patient_path} - {prepoc_raw} - B{target_bvalue}')

        save_plot_path = f"{images_path}{patient_path}_{prepoc_raw}_MSE_dti_b{target_bvalue}.png"

        if save_as_png:
            plt.savefig(save_plot_path)
            print("Saved MSE Plot:", save_plot_path)

        if show:
            plt.show()

        plt.close()

    return 



def process_dti_bval_pairs(patient_path, b0_threshold=60, bet_median_radius=2, bet_numpass=1, bet_dilate=2, prepoc_raw="preproc"):
    """
    Process Diffusion Tensor Imaging (DTI) for an individual patient using pairs of specific b-values.

    This function performs the following steps:
    1. Sets up file paths and directories.
    2. Reads b-values and b-vectors.
    3. Loads the DWI data.
    4. Iterates through pairs of specified b-values and processes the DTI data for each pair.
    5. Performs brain extraction using the median_otsu method.
    6. Fits the DTI model and computes various diffusion metrics (FA, MD, RD, AD).
    7. Saves the computed metrics as NIfTI files.
    8. Computes and saves residuals of the DTI model fit.

    Parameters:
    - patient_path (str): The identifier of the patient.
    - b0_threshold (int): B0 threshold for gradient table estimation. Default is 60.
    - bet_median_radius (int): Median radius for brain extraction. Default is 2.
    - bet_numpass (int): Number of passes for brain extraction. Default is 1.
    - bet_dilate (int): Dilation factor for brain mask. Default is 2.
    - prepoc_raw (str): Specify whether to use preprocessed ("preproc") or raw ("raw") data. Default is "preproc".

    Returns:
    - None
    """
    import datetime
    log_prefix = "DTI SOLO"
    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + f": Beginning of individual DTI processing for patient {patient_path} \n")

    parent_folder = os.path.dirname(os.getcwd())  # Get the parent folder of the current working directory

    dti_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/DTI/"
    images_path = os.path.join(parent_folder, "images/")

    if not os.path.exists(dti_path):
        os.makedirs(dti_path)
    
    print("Folder Path:", parent_folder)
    print("DTI Path:", dti_path)
    print("Images Path:", images_path)
    print("Preprocessing Type:", prepoc_raw)

    fdwi = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/preproc/{patient_path}_dmri_preproc.nii.gz"
    fbval = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/preproc/{patient_path}_dmri_preproc.bval"
    fbvec = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/preproc/{patient_path}_dmri_preproc.bvec"

    print("DWI File Path:", fdwi)
    print("BVAL File Path:", fbval)
    print("BVEC File Path:", fbvec)

    print("Reading b-values and b-vectors...")
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    print("Shape of b-values:", bvals.shape)
    print("Shape of b-vectors:", bvecs.shape)

    print("\nLoading DWI data...")
    data, affine, voxel_size = load_nifti(fdwi, return_voxsize=True)
    print("Shape of DWI data:", data.shape)
    print("Voxel size:", voxel_size)

    specific_bvals = [1000, 3000]
    bval_combinations = [(b1, b2) for i, b1 in enumerate(specific_bvals) for b2 in specific_bvals[i+1:] if b1 != b2]

    print(f"[{log_prefix}] Processing {len(bval_combinations)} combinations of b_values.")

    for bval_pair in bval_combinations:
        target_indices = np.where((abs(bvals - bval_pair[0]) <= 150) | 
                        (abs(bvals - bval_pair[1]) <= 150) | 
                        (bvals == 0))[0]
        
        print(f"Number of elements in target_indices: {len(target_indices)}")

        # Select data and bvecs for the target b-value
        target_data = data[:, :, :, target_indices]
        target_bvals = bvals[target_indices]
        target_bvecs = bvecs[target_indices, :]

        b0_mask, mask = median_otsu(target_data, median_radius=bet_median_radius, numpass=bet_numpass,
                                    vol_idx=range(0, np.shape(target_data)[3]), dilate=bet_dilate)
        
        b0_threshold = np.min(bvals) + 10
        b0_threshold = max(50, b0_threshold)
        gtab = gradient_table(target_bvals, target_bvecs, b0_threshold=b0_threshold)

        tenmodel = dti.TensorModel(gtab)
        tenfit = tenmodel.fit(target_data, mask=mask)

        bval_str = f"b{bval_pair[0]}b{bval_pair[1]}"

        FA = dti.fractional_anisotropy(tenfit.evals)
        FA[np.isnan(FA)] = 0
        FA = np.clip(FA, 0, 1)
        fa_file_path = dti_path + f"{patient_path}_FA_{bval_str}.nii.gz"
        save_nifti(fa_file_path, FA.astype(np.float32), affine)
        print("Saved:", fa_file_path)

        RGB = dti.color_fa(FA, tenfit.evecs)
        rgb_file_path = dti_path + f"{patient_path}_fargb_{bval_str}.nii.gz"
        save_nifti(rgb_file_path, np.array(255 * RGB, 'uint8'), affine)
        print("Saved:", rgb_file_path)

        MD = dti.mean_diffusivity(tenfit.evals)
        md_file_path = dti_path + f"{patient_path}_MD_{bval_str}.nii.gz"
        save_nifti(md_file_path, MD.astype(np.float32), affine)
        print("Saved:", md_file_path)

        RD = dti.radial_diffusivity(tenfit.evals)
        rd_file_path = dti_path + f"{patient_path}_RD_{bval_str}.nii.gz"
        save_nifti(rd_file_path, RD.astype(np.float32), affine)
        print("Saved:", rd_file_path)

        AD = dti.axial_diffusivity(tenfit.evals)
        ad_file_path = dti_path + f"{patient_path}_AD_{bval_str}.nii.gz"
        save_nifti(ad_file_path, AD.astype(np.float32), affine)
        print("Saved:", ad_file_path)

        evecs_file_path = dti_path + f"{patient_path}_evecs_{bval_str}.nii.gz"
        save_nifti(evecs_file_path, tenfit.evecs.astype(np.float32), affine)
        print("Saved:", evecs_file_path)

        evals_file_path = dti_path + f"{patient_path}_evals_{bval_str}.nii.gz"
        save_nifti(evals_file_path, tenfit.evals.astype(np.float32), affine)
        print("Saved:", evals_file_path)

        dtensor_file_path = dti_path + f"{patient_path}_dtensor_{bval_str}.nii.gz"
        save_nifti(dtensor_file_path, tenfit.quadratic_form.astype(np.float32), affine)
        print("Saved:", dtensor_file_path)

        reconstructed = tenfit.predict(gtab, S0=target_data[..., 0])
        residual = target_data - reconstructed
        residual_file_path = dti_path + f"{patient_path}_residual_{bval_str}.nii.gz"
        save_nifti(residual_file_path, residual.astype(np.float32), affine)
        print("Saved:", residual_file_path)

    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + f": DTI processing completed for patient {patient_path}.\n")
        


def calculate_and_plot_dti_mse_pair(patient_path, prepoc_raw="preproc"):
    """
    Calculate and plot Mean Squared Error (MSE) for Diffusion Tensor Imaging (DTI) for pairs of specific b-values.

    This function performs the following steps for each pair of specific b-values:
    1. Sets up file paths and directories.
    2. Reads the residuals of the DTI model fit.
    3. Calculates the Mean Squared Error (MSE) from the residuals.
    4. Prints statistical information about the MSE (mean, median, standard deviation).
    5. Saves the MSE as a NIfTI file.

    Parameters:
    - patient_path (str): The identifier of the patient.
    - prepoc_raw (str): Specify whether to use preprocessed ("preproc") or raw ("raw") data. Default is "preproc".

    Returns:
    - None
    """
    parent_folder = os.path.dirname(os.getcwd())
    dti_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/DTI/"
    images_path = parent_folder + "/images/"

    print("Folder Path:", parent_folder)
    print("DTI Path:", dti_path)
    print("Images Path:", images_path)
    print("Preprocessing Type:", prepoc_raw)

    specific_bvals = [1000, 3000]
    bval_combinations = [(b1, b2) for i, b1 in enumerate(specific_bvals) for b2 in specific_bvals[i+1:] if b1 != b2]

    for bval_pair in bval_combinations:

        bval_str = f"b{bval_pair[0]}b{bval_pair[1]}"
        residual_path = dti_path + patient_path + f"_residual_{bval_str}.nii.gz"
        print("Residual Path:", residual_path)

        residual, affine = load_nifti(residual_path)

        sli = residual.shape[2] // 2

        mse = np.nanmean(residual ** 2, axis=-1)

        print(f"Statistical Information about MSE (DTI - {patient_path} - {prepoc_raw} - {bval_str}):")
        print(f" - Mean: {np.nanmean(mse):.4f}")
        print(f" - Median: {np.nanmedian(mse):.4f}")
        print(f" - Standard Deviation: {np.nanstd(mse):.4f}")

        mse_file_path = dti_path + patient_path + f"_mse_{bval_str}.nii.gz"
        save_nifti(mse_file_path, mse.astype(np.float32), affine)
        print("Saved MSE File:", mse_file_path)


    return

########################################################################################################################################

for i in range(1001, 1011):
    patient_path = f"sub-{i:04d}"

    process_dti(patient_path, b0_threshold=60, bet_median_radius=2, bet_numpass=1, bet_dilate=2)
    calculate_and_plot_dti_mse(patient_path, prepoc_raw="preproc", save_as_png=False, show=False)


    
    process_dti_b_values(patient_path, b0_threshold=60, bet_median_radius=2, bet_numpass=1,bet_dilate=2)
    calculate_and_plot_dti_mse_b_value(patient_path, prepoc_raw="preproc", save_as_png=False, show=True)
    
    
    process_dti_bval_pairs(patient_path, b0_threshold=60, bet_median_radius=2, bet_numpass=1, bet_dilate=2, prepoc_raw="preproc")
    calculate_and_plot_dti_mse_pair(patient_path, prepoc_raw="preproc")

