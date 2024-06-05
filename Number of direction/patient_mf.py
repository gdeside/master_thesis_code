import numpy as np
import datetime
import os 

from dipy.io.image import load_nifti, save_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti
from dipy.segment.mask import median_otsu
from dipy.denoise.localpca import mppca
from dipy.align.reslice import reslice

from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.direction import peaks_from_model
from dipy.data import default_sphere
from dipy.reconst.csdeconv import auto_response_ssst

import microstructure_fingerprinting as mf
import microstructure_fingerprinting.mf_utils as mfu



def process_patient_mf_reduced(patient_path,bval_reduced,nb_vectors):
    """
    Process individual microstructure fingerprinting for a given patient.

    Parameters:
    - patient_path (str): The name of the patient.

    Returns:
    - None
    """

    log_prefix = "MF SOLO"
    current_time = datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S")
    print(f"[{log_prefix}] {current_time}: Starting processing for patient {patient_path} with bval_reduced {bval_reduced} and nb_vectors {nb_vectors}.")

    folder_path = os.path.dirname(os.getcwd()) 

    images_path = os.path.join(folder_path, "images")
    mf_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/"

    fdwi = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/preproc/{patient_path}_dmri_preproc.nii.gz"
    fbval = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/preproc/{patient_path}_dmri_preproc.bval"
    fbvec = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/preproc/{patient_path}_dmri_preproc.bvec"

    print(f"[{log_prefix}] DWI: {fdwi}")
    print(f"[{log_prefix}] BVAL: {fbval}")
    print(f"[{log_prefix}] BVEC: {fbvec}")
    print("[" + log_prefix + "] File paths configured successfully.")

    dictionary_path = "/auto/home/users/g/d/gdeside/Mine/dictionaries/dictionary-fixedraddist_scheme-HCPMGH.mat"

    data, affine = load_nifti(fdwi)
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)

    input_file_path = f"/auto/home/users/g/d/gdeside/Mine/Spheres/{patient_path}/{bval_reduced}/{patient_path}_closest_points_full_b{bval_reduced}_{nb_vectors}.txt"

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


    # Verify the shapes
    print(f"Shape of reduced bvals: {updated_bvals.shape}")
    print(f"Shape of reduced data: {updated_data.shape}")

    print("[" + log_prefix + "] Data loaded and gradient table prepared.")

    b0_threshold = np.min(updated_bvals) + 10
    b0_threshold = max(50, b0_threshold)

    data_CSD = updated_data
    gtab_CSD = gradient_table(updated_bvals, updated_bvecs, b0_threshold=b0_threshold)

    bet_median_radius = 2
    bet_numpass = 1
    bet_dilate = 2

    b0_mask, mask = median_otsu(updated_data, median_radius=bet_median_radius, numpass=bet_numpass, vol_idx=range(0, np.shape(updated_data)[3]), dilate=bet_dilate)

    response, ratio = auto_response_ssst(gtab_CSD, data_CSD, roi_radii=10, fa_thr=0.7)

    csd_model = ConstrainedSphericalDeconvModel(gtab_CSD, response, sh_order=8)
    csd_peaks = peaks_from_model(npeaks=2, model=csd_model, data=data_CSD, sphere=default_sphere,
                                    relative_peak_threshold=0.25, min_separation_angle=25, parallel=True, mask=mask,
                                    normalize_peaks=True, return_odf=False, return_sh=True)

    numfasc_2 = np.sum(csd_peaks.peak_values[:, :, :, 0] > 0.15) + np.sum(
                csd_peaks.peak_values[:, :, :, 1] > 0.15)

    normPeaks0 = csd_peaks.peak_dirs[..., 0, :]
    normPeaks1 = csd_peaks.peak_dirs[..., 1, :]
    for i in range(np.shape(csd_peaks.peak_dirs)[0]):
            for j in range(np.shape(csd_peaks.peak_dirs)[1]):
                for k in range(np.shape(csd_peaks.peak_dirs)[2]):
                    norm = np.sqrt(np.sum(normPeaks0[i, j, k, :] ** 2))
                    normPeaks0[i, j, k, :] = normPeaks0[i, j, k, :] / norm
                    norm = np.sqrt(np.sum(normPeaks1[i, j, k, :] ** 2))
                    normPeaks1[i, j, k, :] = normPeaks1[i, j, k, :] / norm
    mu1 = normPeaks0
    mu2 = normPeaks1
    frac1 = csd_peaks.peak_values[..., 0]
    frac2 = csd_peaks.peak_values[..., 1]
    (peaks, numfasc) = mf.cleanup_2fascicles(frac1=frac1, frac2=frac2,
                                                    mu1=mu1, mu2=mu2,
                                                    peakmode='peaks',
                                                    mask=mask, frac12=None)

    mf_model = mf.MFModel(dictionary_path)
    csf_mask = True
    ear_mask = False
    MF_fit = mf_model.fit(updated_data, mask, numfasc, peaks=peaks, bvals=updated_bvals,
                              bvecs=updated_bvecs, csf_mask=csf_mask, ear_mask=ear_mask,
                              verbose=3, parallel=True, cpu_counts=20)

    frac_f0 = MF_fit.frac_f0
    fvf_tot = MF_fit.fvf_tot
    MSE = MF_fit.MSE
    R2 = MF_fit.R2

    print("[" + log_prefix + "] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": MF model fitting completed.", flush=True)

    MF_fit.write_nifti(os.path.join(mf_path,f'{patient_path}_reduced_b{bval_reduced}_{nb_vectors}.nii.gz'), affine=affine)

    print("[" + log_prefix + "] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Process completed for patient %s" % patient_path, flush=True)

########################################################################################################################################################################
for i in range(1007, 1008):
    
    patient_path = f"sub-{i:04d}"
    print(patient_path)

    #process_patient_mf_reduced(patient_path,1000,48)

    #process_patient_mf_reduced(patient_path,3000,16)
    #process_patient_mf_reduced(patient_path,3000,32)
    #process_patient_mf_reduced(patient_path,3000,40)
    #process_patient_mf_reduced(patient_path,3000,48)

    process_patient_mf_reduced(patient_path,5000,64)
    #process_patient_mf_reduced(patient_path,5000,100)

    #process_patient_mf_reduced(patient_path,10000,128)
    #process_patient_mf_reduced(patient_path,10000,200)
