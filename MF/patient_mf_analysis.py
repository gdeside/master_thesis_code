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




def process_patient_mf(patient_path):
    """
    Process individual microstructure fingerprinting for a given patient.

    This function performs the following steps to process microstructure fingerprinting data:
    1. Load DWI data, b-values, and b-vectors.
    2. Prepare a gradient table and perform brain extraction.
    3. Fit a Constrained Spherical Deconvolution (CSD) model to the data.
    4. Extract peak directions and normalize them.
    5. Clean up peaks to identify fascicles.
    6. Fit the Microstructure Fingerprinting (MF) model.
    7. Calculate and save the MF metrics (fraction of free water, total fiber volume fraction, MSE, and R2).

    Parameters:
    - patient_path (str): The identifier of the patient.

    Returns:
    - None
    """
    log_prefix = "MF SOLO"
    print("[" + log_prefix + "] " + datetime.datetime.now().strftime(
        "%d.%b %Y %H:%M:%S") + ": Beginning of individual microstructure fingerprinting processing for patient %s \n" % patient_path, flush=True)

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

    print("[" + log_prefix + "] Data loaded and gradient table prepared.")

    b0_threshold = np.min(bvals) + 10
    b0_threshold = max(50, b0_threshold)

    data_CSD = data
    gtab_CSD = gradient_table(bvals, bvecs, b0_threshold=b0_threshold)

    bet_median_radius = 2
    bet_numpass = 1
    bet_dilate = 2

    b0_mask, mask = median_otsu(data, median_radius=bet_median_radius, numpass=bet_numpass, vol_idx=range(0, np.shape(data)[3]), dilate=bet_dilate)

    response, ratio = auto_response_ssst(gtab_CSD, data_CSD, roi_radii=10, fa_thr=0.7)

    csd_model = ConstrainedSphericalDeconvModel(gtab_CSD, response, sh_order=8)
    csd_peaks = peaks_from_model(npeaks=2, model=csd_model, data=data_CSD, sphere=default_sphere,
                                    relative_peak_threshold=0.25, min_separation_angle=25, parallel=False, mask=mask,
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
    MF_fit = mf_model.fit(data, mask, numfasc, peaks=peaks, bvals=bvals,
                              bvecs=bvecs, csf_mask=csf_mask, ear_mask=ear_mask,
                              verbose=3, parallel=True, cpu_counts=20)

    frac_f0 = MF_fit.frac_f0
    fvf_tot = MF_fit.fvf_tot
    MSE = MF_fit.MSE
    R2 = MF_fit.R2

    print("[" + log_prefix + "] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": MF model fitting completed.", flush=True)

    MF_fit.write_nifti(os.path.join(mf_path,f'{patient_path}.nii.gz'), affine=affine)

    print("[" + log_prefix + "] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Process completed for patient %s" % patient_path, flush=True)


def process_patient_mf_unique_bvals(patient_path):
    """
    Process individual microstructure fingerprinting for a given patient
    focusing on specific b-values in the bvals file.

    This function performs the following steps for each specified b-value:
    1. Load DWI data, b-values, and b-vectors.
    2. Identify and process data for specific b-values (1000, 3000, 5000).
    3. Prepare a gradient table and perform brain extraction using median Otsu.
    4. Fit a Constrained Spherical Deconvolution (CSD) model to the data.
    5. Extract and normalize peak directions, then clean up peaks to identify fascicles.
    6. Fit the Microstructure Fingerprinting (MF) model.
    7. Calculate and save the MF metrics (fraction of free water, total fiber volume fraction, MSE, and R2).

    Parameters:
    - patient_path (str): The identifier of the patient.

    Returns:
    - None
    """
    log_prefix = "MF UNIQUE BVALS"
    print(f"[{log_prefix}] {datetime.datetime.now().strftime('%d.%b %Y %H:%M:%S')}: "
          f"Beginning of processing for patient {patient_path} with unique b_values analysis\n", flush=True)

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

    # Identify unique b_values
    unique_bvals = np.unique(bvals)
    specific_bvals = [1000, 3000, 5000]
    print(f"[{log_prefix}] Identified unique b_values: {unique_bvals}")

    for bval in specific_bvals:
        print(f"[{log_prefix}] Processing for b_value: {bval}")
        bval_indices = np.where((abs(bvals - bval) <= 150) | (bvals == 0))[0]
        print(f"[{log_prefix}] Size of bval_indices: {len(bval_indices)}")


        if len(bval_indices) > 0:
            data_for_bval = data[..., bval_indices]
            bvals_for_bval = bvals[bval_indices]
            bvecs_for_bval =bvecs[bval_indices]

            b0_threshold = np.min(bvals_for_bval) + 10
            b0_threshold = max(50, b0_threshold)

            gtab_CSD = gradient_table(bvals_for_bval, bvecs_for_bval, b0_threshold=b0_threshold)

            bet_median_radius = 2
            bet_numpass = 1
            bet_dilate = 2

            b0_mask, mask = median_otsu(data_for_bval, median_radius=bet_median_radius, numpass=bet_numpass, vol_idx=range(0, np.shape(data_for_bval)[3]), dilate=bet_dilate)

            response, ratio = auto_response_ssst(gtab_CSD, data_for_bval, roi_radii=10, fa_thr=0.7)

            csd_model = ConstrainedSphericalDeconvModel(gtab_CSD, response, sh_order=8)
            csd_peaks = peaks_from_model(npeaks=2, model=csd_model, data=data_for_bval, sphere=default_sphere,
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
            MF_fit = mf_model.fit(data_for_bval, mask, numfasc, peaks=peaks, bvals=bvals_for_bval,
                                    bvecs=bvecs_for_bval, csf_mask=csf_mask, ear_mask=ear_mask,
                                    verbose=3, parallel=True, cpu_counts=20)

            frac_f0 = MF_fit.frac_f0
            fvf_tot = MF_fit.fvf_tot
            MSE = MF_fit.MSE
            R2 = MF_fit.R2

            print("[" + log_prefix + "] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": MF model fitting completed.", flush=True)

            MF_fit.write_nifti(os.path.join(mf_path,f'{patient_path}_b{bval}.nii.gz'), affine=affine)

            print("[" + log_prefix + "] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Process completed for patient %s" % patient_path, flush=True)




def process_patient_mf_bval_pairs(patient_path):
    """
    Process individual microstructure fingerprinting for a given patient using pairs of specific b-values.

    This function performs the following steps for each pair of specified b-values:
    1. Load DWI data, b-values, and b-vectors.
    2. Identify and process data for specific b-value pairs (1000, 3000, 5000, 10000).
    3. Prepare a gradient table and load the brain mask.
    4. Perform brain extraction using the provided brain mask.
    5. Fit a Constrained Spherical Deconvolution (CSD) model to the data.
    6. Extract and normalize peak directions, then clean up peaks to identify fascicles.
    7. Fit the Microstructure Fingerprinting (MF) model.
    8. Calculate and save the MF metrics (fraction of free water, total fiber volume fraction, MSE, and R2).

    Parameters:
    - patient_path (str): The identifier of the patient.

    Returns:
    - None
    """
    log_prefix = "MF UNIQUE BVALS"
    print(f"[{log_prefix}] {datetime.datetime.now().strftime('%d.%b %Y %H:%M:%S')}: "
          f"Beginning of processing for patient {patient_path} with pair b_values analysis\n", flush=True)

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

    # Identify unique b_values
    unique_bvals = np.unique(bvals)
    specific_bvals = [1000, 3000, 5000, 10000]
    bval_combinations = [(b1, b2) for i, b1 in enumerate(specific_bvals) for b2 in specific_bvals[i+1:] if b1 != b2]

    print(f"[{log_prefix}] Identified unique b_values: {unique_bvals}")
    print(f"[{log_prefix}] Processing {len(bval_combinations)} combinations of b_values.")

    for bval_pair in bval_combinations:
         bval_indices = np.where((abs(bvals - bval_pair[0]) <= 150) | 
                        (abs(bvals - bval_pair[1]) <= 150) | 
                        (bvals == 0))[0]
         
         print(f"[{log_prefix}] Size of bval_indices: {len(bval_indices)}")
        
        
         if len(bval_indices) > 0:
            data_for_bval = data[..., bval_indices]
            bvals_for_bval = bvals[bval_indices]
            bvecs_for_bval =bvecs[bval_indices]

            b0_threshold = np.min(bvals_for_bval) + 10
            b0_threshold = max(50, b0_threshold)

            gtab_CSD = gradient_table(bvals_for_bval, bvecs_for_bval, b0_threshold=b0_threshold)

            bet_median_radius = 2
            bet_numpass = 1
            bet_dilate = 2

            # Load mask
            mask,_ = load_nifti(f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/masks/{patient_path}_brain_mask_dilated.nii.gz")

            response, ratio = auto_response_ssst(gtab_CSD, data_for_bval, roi_radii=10, fa_thr=0.7)

            csd_model = ConstrainedSphericalDeconvModel(gtab_CSD, response, sh_order=8)
            csd_peaks = peaks_from_model(npeaks=2, model=csd_model, data=data_for_bval, sphere=default_sphere,
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
            MF_fit = mf_model.fit(data_for_bval, mask, numfasc, peaks=peaks, bvals=bvals_for_bval,
                                    bvecs=bvecs_for_bval, csf_mask=csf_mask, ear_mask=ear_mask,
                                    verbose=3, parallel=True, cpu_counts=25)

            frac_f0 = MF_fit.frac_f0
            fvf_tot = MF_fit.fvf_tot
            MSE = MF_fit.MSE
            R2 = MF_fit.R2

            print("[" + log_prefix + "] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": MF model fitting completed.", flush=True)

            bval_str = f"b{bval_pair[0]}b{bval_pair[1]}"  # This creates a string like "b1000b3000"
            MF_fit.write_nifti(os.path.join(mf_path, f'{patient_path}_{bval_str}.nii.gz'), affine=affine)

    print("[" + log_prefix + "] " + datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S") + ": Process completed for patient %s" % patient_path, flush=True)

         
        








####################################################################################################################################
for i in range(1001, 1011):
    patient_path = f"sub-{i:04d}"
    process_patient_mf(patient_path)
    process_patient_mf_unique_bvals(patient_path)
    process_patient_mf_bval_pairs(patient_path)
