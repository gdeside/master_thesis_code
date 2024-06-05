import numpy as np
import datetime
import os
from dipy.io.image import load_nifti, save_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti
from dipy.segment.mask import median_otsu
import matplotlib.pyplot as plt
import copy

def process_dti_reduce_bvalues(patient_path, bval_reduced, nb_vectors,b0_threshold=60, bet_median_radius=2, bet_numpass=1, bet_dilate=2, prepoc_raw="preproc"):
    """
    Process Diffusion Tensor Imaging (DTI) data for an individual patient with reduced b-values.

    This function performs the following steps:
    1. Load DWI data, b-values, and b-vectors.
    2. Read and use new b-vectors from a specified file for a specific b-value.
    3. Identify and process data for reduced b-values.
    4. Prepare a gradient table and perform brain extraction using median Otsu.
    5. Fit a Tensor model to the data.
    6. Calculate and save DTI metrics (FA, MD, RD, AD), color FA, eigenvalues, eigenvectors, diffusion tensor, and residuals.

    Parameters:
    - patient_path (str): The identifier of the patient.
    - bval_reduced (int): The specific b-value to reduce and process.
    - nb_vectors (int): The number of vectors to be used for the specific b-value.
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

    input_file_path = f"/auto/home/users/g/d/gdeside/Mine/Spheres/{patient_path}/{bval_reduced}/{patient_path}_closest_points_full_b{bval_reduced}_{nb_vectors}.txt"

    # Load existing bvals and bvecs
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    data, affine, voxel_size = load_nifti(fdwi, return_voxsize=True)


    start_end_index_dic = {
    0 : [0,40] , 
    1000 : [40,104] ,
    3000 : [104,168] ,
    5000 : [168,296] ,
    10000 : [296,552] ,
    }

    start_index = start_end_index_dic[bval_reduced][0]
    end_index   = start_end_index_dic[bval_reduced][1]

    # Load new_bvecs for bval = 5000 from the saved file
    new_bvecs = np.loadtxt(input_file_path, delimiter=',')
    new_bvecs = new_bvecs[start_index:start_index+nb_vectors]

    # Identify indices for bval = 5000 and other bvals
    indices_b5000 = np.where(bvals == bval_reduced)[0]
    indices_not_b5000 = np.where(bvals != bval_reduced)[0]

    # Initialize lists for updated bvals and bvecs
    updated_bvals = []
    updated_bvecs = []
    updated_data_indices = []

    # Add all bvecs and bvals not equal to 5000
    for idx in indices_not_b5000:
        updated_bvals.append(bvals[idx])
        updated_bvecs.append(bvecs[idx])
        updated_data_indices.append(idx)

    used_indices_new_bvecs = set()

    # For bvals equal to 5000, add only those that match new_bvecs
    bvecs_b5000 = bvecs[indices_b5000]
    for idx, bv in enumerate(new_bvecs):
        print(idx, bv)
        distances = np.linalg.norm(bvecs_b5000 - bv, axis=1)  # Compute the Euclidean distance
        order_of_indices = np.argsort(distances)
        
        # Find the closest non-used vector
        for new_idx in order_of_indices:
            if new_idx not in used_indices_new_bvecs:
                closest_vector = bvecs_b5000[new_idx]
                used_indices_new_bvecs.add(indices_b5000[new_idx])

                print(f"Matching new_bvecs index {idx} with bvecs_b5000 index {new_idx} (global index {indices_b5000[new_idx]})")
            
                break
        
        print(f"Closest vector from new_bvecs (Index {idx}): {bv}\nMatched with original vector from bvecs_b5000 (Global Index {indices_b5000[new_idx]}, Local Index {new_idx}): {bvecs[indices_b5000[new_idx]]}")
        updated_bvals.append(bvals[indices_b5000[new_idx]])
        updated_bvecs.append(closest_vector)
        updated_data_indices.append(indices_b5000[new_idx])

    # Convert updated lists to arrays
    updated_bvals = np.array(updated_bvals)
    updated_bvecs = np.array(updated_bvecs)

    # Sort the updated data indices and use them to select the corresponding slices from the data
    #updated_data_indices.sort()  # Ensure the data is selected in the correct order
    print(updated_data_indices)
    updated_data = data[..., updated_data_indices]


    target_indices = np.where((abs(updated_bvals - 1000) <= 150) | (updated_bvals == 0) | (abs(updated_bvals - 3000) <= 150) )[0]


    # Select data and bvecs for the target b-value
    updated_data = updated_data[:, :, :, target_indices]
    updated_bvals = updated_bvals[target_indices]
    updated_bvecs = updated_bvecs[target_indices, :]


    # Verify the shapes
    print(f"Shape of reduced bvals: {updated_bvals.shape}")
    print(f"Shape of reduced data: {updated_data.shape}")

    b0_mask, mask = median_otsu(updated_data, median_radius=bet_median_radius, numpass=bet_numpass,
                                vol_idx=range(0, np.shape(updated_data)[3]), dilate=bet_dilate)

    b0_threshold = np.min(updated_bvals) + 10
    b0_threshold = max(50, b0_threshold)
    gtab = gradient_table(updated_bvals, updated_bvecs, b0_threshold=b0_threshold)

    tenmodel = dti.TensorModel(gtab)
    tenfit = tenmodel.fit(updated_data, mask=mask)

    FA = dti.fractional_anisotropy(tenfit.evals)
    FA[np.isnan(FA)] = 0
    FA = np.clip(FA, 0, 1)

    base_file_path = dti_path + patient_path + f"_reduced_b{bval_reduced}_{nb_vectors}_"

    # Fractional Anisotropy
    fa_file_path = base_file_path + "FA.nii.gz"
    save_nifti(fa_file_path, FA.astype(np.float32), affine)
    print("Saved:", fa_file_path)

    # Color FA
    RGB = dti.color_fa(FA, tenfit.evecs)
    rgb_file_path = base_file_path + "fargb.nii.gz"
    save_nifti(rgb_file_path, np.array(255 * RGB, 'uint8'), affine)
    print("Saved:", rgb_file_path)

    # Mean Diffusivity
    MD = dti.mean_diffusivity(tenfit.evals)
    md_file_path = base_file_path + "MD.nii.gz"
    save_nifti(md_file_path, MD.astype(np.float32), affine)
    print("Saved:", md_file_path)

    # Radial Diffusivity
    RD = dti.radial_diffusivity(tenfit.evals)
    rd_file_path = base_file_path + "RD.nii.gz"
    save_nifti(rd_file_path, RD.astype(np.float32), affine)
    print("Saved:", rd_file_path)

    # Axial Diffusivity
    AD = dti.axial_diffusivity(tenfit.evals)
    ad_file_path = base_file_path + "AD.nii.gz"
    save_nifti(ad_file_path, AD.astype(np.float32), affine)
    print("Saved:", ad_file_path)

    # Eigen Vectors
    evecs_file_path = base_file_path + "evecs.nii.gz"
    save_nifti(evecs_file_path, tenfit.evecs.astype(np.float32), affine)
    print("Saved:", evecs_file_path)

    # Eigen Values
    evals_file_path = base_file_path + "evals.nii.gz"
    save_nifti(evals_file_path, tenfit.evals.astype(np.float32), affine)
    print("Saved:", evals_file_path)

    # Diffusion Tensor
    dtensor_file_path = base_file_path + "dtensor.nii.gz"
    save_nifti(dtensor_file_path, tenfit.quadratic_form.astype(np.float32), affine)
    print("Saved:", dtensor_file_path)

    # Residuals from model fitting
    reconstructed = tenfit.predict(gtab, S0=updated_data[..., 0])
    residual = updated_data - reconstructed
    residual_file_path = base_file_path + "residual.nii.gz"
    save_nifti(residual_file_path, residual.astype(np.float32), affine)
    print("Saved:", residual_file_path)

    print("[" + log_prefix + "] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + f": DTI processing completed for patient {patient_path} with reduced b-vectors.\n")

    return 


def calculate_and_plot_dti_reduced_mse(patient_path, bval_reduced, nb_vectors, prepoc_raw="preproc", save_as_png=False, show=False):
    """
    Calculate and plot Mean Squared Error (MSE) for Diffusion Tensor Imaging (DTI) with reduced b-values.

    This function performs the following steps:
    1. Sets up file paths and directories.
    2. Loads the residuals of the DTI model fit for the reduced b-values.
    3. Calculates the Mean Squared Error (MSE) from the residuals.
    4. Prints statistical information about the MSE (mean, median, standard deviation).
    5. Saves the MSE as a NIfTI file.
    6. Plots the MSE and optionally saves the plot as a PNG file.

    Parameters:
    - patient_path (str): The identifier of the patient.
    - bval_reduced (int): The specific b-value that has been reduced and processed.
    - nb_vectors (int): The number of vectors used for the specific reduced b-value.
    - prepoc_raw (str, optional): Specify whether to use preprocessed ("preproc") or raw ("raw") data. Default is "preproc".
    - save_as_png (bool, optional): Whether to save the plot as a PNG file. Default is False.
    - show (bool, optional): Whether to display the plot using plt.show(). Default is False.

    Returns:
    - mse (numpy.ndarray): The calculated MSE for the reduced DTI data.
    """
    folder_path = os.path.dirname(os.getcwd())
    dti_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/DTI/"
    images_path = folder_path + "/images/"

    print("Folder Path:", folder_path)
    print("DTI Path:", dti_path)
    print("Images Path:", images_path)
    print("Preprocessing Type:", prepoc_raw)

    residual_path = dti_path + patient_path + f"_reduced_b{bval_reduced}_{nb_vectors}_residual.nii.gz"
    print("Residual Path:", residual_path)

    residual, affine = load_nifti(residual_path)

    sli = residual.shape[2] // 2

    mse = np.nanmean(residual ** 2, axis=-1)

    print(f"Statistical Information about MSE (DTI - {patient_path} - {prepoc_raw}):")
    print(f" - Mean: {np.nanmean(mse):.4f}")
    print(f" - Median: {np.nanmedian(mse):.4f}")
    print(f" - Standard Deviation: {np.nanstd(mse):.4f}")


    mse_file_path = dti_path + patient_path + f"_reduced_b{bval_reduced}_{nb_vectors}_mse.nii.gz"
    save_nifti(mse_file_path, mse.astype(np.float32), affine)
    print("Saved MSE File:", mse_file_path)

    # Plot MSE
    plt.figure(f'MSE DTI - {patient_path} - {prepoc_raw}')
    plt.imshow(mse[:, :, sli])
    plt.title(f'MSE DTI - {patient_path} - {prepoc_raw}')

    save_plot_path = f"{images_path}{patient_path}_{prepoc_raw}_reduced_MSE_dti.png"

    if save_as_png:
        plt.savefig(save_plot_path)
        print("Saved MSE Plot:", save_plot_path)

    if show:
        plt.show()

    return mse

################################################################

for i in range(1001, 1011):
    patient_path = f"sub-{i:04d}"

    process_dti_reduce_bvalues(patient_path,1000, 16, b0_threshold=60, bet_median_radius=2, bet_numpass=1, bet_dilate=2, prepoc_raw="preproc")
    calculate_and_plot_dti_reduced_mse(patient_path,1000, 16, prepoc_raw="preproc", save_as_png=False, show=False)

    process_dti_reduce_bvalues(patient_path,1000, 32, b0_threshold=60, bet_median_radius=2, bet_numpass=1, bet_dilate=2, prepoc_raw="preproc")
    calculate_and_plot_dti_reduced_mse(patient_path,1000, 32, prepoc_raw="preproc", save_as_png=False, show=False)

    process_dti_reduce_bvalues(patient_path,1000, 40, b0_threshold=60, bet_median_radius=2, bet_numpass=1, bet_dilate=2, prepoc_raw="preproc")
    calculate_and_plot_dti_reduced_mse(patient_path,1000, 40, prepoc_raw="preproc", save_as_png=False, show=False)

    process_dti_reduce_bvalues(patient_path,1000, 48, b0_threshold=60, bet_median_radius=2, bet_numpass=1, bet_dilate=2, prepoc_raw="preproc")
    calculate_and_plot_dti_reduced_mse(patient_path,1000, 48, prepoc_raw="preproc", save_as_png=False, show=False)