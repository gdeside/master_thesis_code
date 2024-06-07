import numpy as np
import os
import datetime
from dipy.io.image import load_nifti, save_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dmipy.core.acquisition_scheme import gtab_dipy2dmipy
from dmipy.signal_models import cylinder_models, gaussian_models
from dmipy.distributions.distribute_models import SD1WatsonDistributed
from dmipy.core.modeling_framework import MultiCompartmentModel

from dipy.segment.mask import median_otsu

from itertools import combinations

def process_noddi_data(patient_path, preproc="preproc",b0_threshold=60, use_parallel_processing=True, number_of_processors=None):
    """
    Process NODDI data for a given patient.

    Parameters:
    - patient_path (str): The name of the patient.
    - preproc (str): Type of preprocessing. Default is "preproc".
    - use_parallel_processing (bool): Whether to use parallel processing. Default is True.
    - number_of_processors (int): Number of processors to use for parallel processing. Default is None.

    Returns:
    - None
    """
    log_prefix = "NODDI Processing"
    print(f"[{log_prefix}] {datetime.datetime.now().strftime('%d.%b %Y %H:%M:%S')}: "
          f"Beginning processing for patient {patient_path}\n", flush=True)

    # Set up paths
    folder_path = os.path.dirname(os.getcwd())
    print(f"[{log_prefix}] Setting up paths...")
    noddi_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/noddi/"
    print(f"[{log_prefix}] NODDI path: {noddi_path}")

    # Check if the NODDI path exists, and create it if it doesn't
    if not os.path.exists(noddi_path):
      os.makedirs(noddi_path)
      print(f"[{log_prefix}] Created NODDI path: {noddi_path}")


    fdwi = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/preproc/{patient_path}_dmri_preproc.nii.gz"
    fbval = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/preproc/{patient_path}_dmri_preproc.bval"
    fbvec = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/preproc/{patient_path}_dmri_preproc.bvec"

    print(f"[{log_prefix}] DWI: {fdwi}")
    print(f"[{log_prefix}] BVAL: {fbval}")
    print(f"[{log_prefix}] BVEC: {fbvec}")

    # Load data
    print(f"[{log_prefix}] Loading DWI data...")
    data, affine = load_nifti(fdwi)
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)


    # Define NODDI models
    ball = gaussian_models.G1Ball()
    stick = cylinder_models.C1Stick()
    zeppelin = gaussian_models.G2Zeppelin()

    watson_dispersed_bundle = SD1WatsonDistributed(models=[stick, zeppelin])
    watson_dispersed_bundle.set_tortuous_parameter('G2Zeppelin_1_lambda_perp', 'C1Stick_1_lambda_par', 'partial_volume_0')
    watson_dispersed_bundle.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
    watson_dispersed_bundle.set_fixed_parameter('G2Zeppelin_1_lambda_par', 1.7e-9)

    NODDI_mod = MultiCompartmentModel(models=[ball, watson_dispersed_bundle])
    NODDI_mod.set_fixed_parameter('G1Ball_1_lambda_iso', 3.e-9)

    # Load mask
    mask,_ = load_nifti(f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/masks/{patient_path}_brain_mask_dilated.nii.gz")


    # Transform bval, bvecs
    b0_threshold = np.min(bvals) + 10
    b0_threshold = max(50, b0_threshold)
    gtab_dipy = gradient_table(bvals, bvecs, b0_threshold=b0_threshold)
    acq_scheme_dmipy = gtab_dipy2dmipy(gtab_dipy, b0_threshold=b0_threshold*1e6)
    
    # Fit NODDI model
    NODDI_fit = NODDI_mod.fit(acq_scheme_dmipy, data, mask=mask,
                               use_parallel_processing=use_parallel_processing,
                               number_of_processors=number_of_processors)

    # Get fitted parameters
    fitted_parameters = NODDI_fit.fitted_parameters
    mu = fitted_parameters["SD1WatsonDistributed_1_SD1Watson_1_mu"]
    odi = fitted_parameters["SD1WatsonDistributed_1_SD1Watson_1_odi"]
    f_iso = fitted_parameters["partial_volume_0"]
    f_bundle = fitted_parameters["partial_volume_1"]
    f_intra = (fitted_parameters['SD1WatsonDistributed_1_partial_volume_0'] * fitted_parameters['partial_volume_1'])
    f_icvf = fitted_parameters['SD1WatsonDistributed_1_partial_volume_0'] * (fitted_parameters['partial_volume_1'] > 0.05)
    f_extra = ((1 - fitted_parameters['SD1WatsonDistributed_1_partial_volume_0']) * fitted_parameters['partial_volume_1'])

    # Calculate MSE and R2
    mse = NODDI_fit.mean_squared_error(data)
    R2 = NODDI_fit.R2_coefficient_of_determination(data)

    # Save results
    save_nifti(os.path.join(noddi_path, f'{patient_path}_mu.nii.gz'), mu.astype(np.float32), affine)
    save_nifti(os.path.join(noddi_path, f'{patient_path}_odi.nii.gz'), odi.astype(np.float32), affine)
    save_nifti(os.path.join(noddi_path, f'{patient_path}_fiso.nii.gz'), f_iso.astype(np.float32), affine)
    save_nifti(os.path.join(noddi_path, f'{patient_path}_fbundle.nii.gz'), f_bundle.astype(np.float32), affine)
    save_nifti(os.path.join(noddi_path, f'{patient_path}_fintra.nii.gz'), f_intra.astype(np.float32), affine)
    save_nifti(os.path.join(noddi_path, f'{patient_path}_icvf.nii.gz'), f_icvf.astype(np.float32), affine)
    save_nifti(os.path.join(noddi_path, f'{patient_path}_fextra.nii.gz'), f_extra.astype(np.float32), affine)
    save_nifti(os.path.join(noddi_path, f'{patient_path}_mse.nii.gz'), mse.astype(np.float32), affine)
    save_nifti(os.path.join(noddi_path, f'{patient_path}_R2.nii.gz'), R2.astype(np.float32), affine)

    print(f"[{log_prefix}] {datetime.datetime.now().strftime('%d.%b %Y %H:%M:%S')}: "
          f"NODDI processing completed for patient {patient_path}\n", flush=True)
    


def process_noddi_data_per_bval(patient_path, preproc="preproc",b0_threshold=60, use_parallel_processing=True, number_of_processors=None):
    """
    Process NODDI data for a given patient for each unique b-value separately.

    Parameters:
    - patient_path (str): The name of the patient.
    - preproc (str): Type of preprocessing. Default is "preproc".
    - use_parallel_processing (bool): Whether to use parallel processing. Default is True.
    - number_of_processors (int): Number of processors to use for parallel processing. Default is None.

    Returns:
    - None
    """
    log_prefix = "NODDI Processing Per B-Value"
    print(f"[{log_prefix}] {datetime.datetime.now().strftime('%d.%b %Y %H:%M:%S')}: "
          f"Beginning processing for patient {patient_path}\n", flush=True)

    # Set up paths
    folder_path = os.path.dirname(os.getcwd())
    print(f"[{log_prefix}] Setting up paths...")
    noddi_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/noddi/"
    print(f"[{log_prefix}] NODDI path: {noddi_path}")

    fdwi = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/preproc/{patient_path}_dmri_preproc.nii.gz"
    fbval = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/preproc/{patient_path}_dmri_preproc.bval"
    fbvec = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/preproc/{patient_path}_dmri_preproc.bvec"
    
    print(f"[{log_prefix}] DWI: {fdwi}")
    print(f"[{log_prefix}] BVAL: {fbval}")
    print(f"[{log_prefix}] BVEC: {fbvec}")

    # Load data
    data, affine = load_nifti(fdwi)
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)

    # Identify unique b-values
    unique_bvals = np.unique(bvals)
    specific_bvals =[1000, 3000, 5000, 10000]

    for bval in specific_bvals:
        print(f"[{log_prefix}] Processing for b-value: {bval}")
        bval_indices = np.where((abs(bvals - bval) <= 150) | (bvals == 0))[0]

        if len(bval_indices) > 0:
            # Select data for the current b-value
            data_for_bval = data[..., bval_indices]
            bvals_for_bval = bvals[bval_indices]
            bvecs_for_bval = bvecs[bval_indices]

            # Define NODDI models
            ball = gaussian_models.G1Ball()
            stick = cylinder_models.C1Stick()
            zeppelin = gaussian_models.G2Zeppelin()

            watson_dispersed_bundle = SD1WatsonDistributed(models=[stick, zeppelin])
            watson_dispersed_bundle.set_tortuous_parameter('G2Zeppelin_1_lambda_perp', 'C1Stick_1_lambda_par', 'partial_volume_0')
            watson_dispersed_bundle.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
            watson_dispersed_bundle.set_fixed_parameter('G2Zeppelin_1_lambda_par', 1.7e-9)

            NODDI_mod = MultiCompartmentModel(models=[ball, watson_dispersed_bundle])
            NODDI_mod.set_fixed_parameter('G1Ball_1_lambda_iso', 3.e-9)

            # Load mask
            mask,_ = load_nifti(f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/masks/{patient_path}_brain_mask_dilated.nii.gz")

            # Transform bval, bvecs
            b0_threshold = np.min(bvals_for_bval) + 10
            b0_threshold = max(50, b0_threshold)
            gtab_dipy = gradient_table(bvals_for_bval, bvecs_for_bval, b0_threshold=b0_threshold)
            acq_scheme_dmipy = gtab_dipy2dmipy(gtab_dipy, b0_threshold=b0_threshold*1e6)

            # Fit NODDI model
            NODDI_fit = NODDI_mod.fit(acq_scheme_dmipy, data_for_bval, mask=mask,
                                       use_parallel_processing=use_parallel_processing,
                                       number_of_processors=number_of_processors)

            # Get fitted parameters
            fitted_parameters = NODDI_fit.fitted_parameters
            mu = fitted_parameters["SD1WatsonDistributed_1_SD1Watson_1_mu"]
            odi = fitted_parameters["SD1WatsonDistributed_1_SD1Watson_1_odi"]
            f_iso = fitted_parameters["partial_volume_0"]
            f_bundle = fitted_parameters["partial_volume_1"]
            f_intra = (fitted_parameters['SD1WatsonDistributed_1_partial_volume_0'] * fitted_parameters['partial_volume_1'])
            f_icvf = fitted_parameters['SD1WatsonDistributed_1_partial_volume_0'] * (fitted_parameters['partial_volume_1'] > 0.05)
            f_extra = ((1 - fitted_parameters['SD1WatsonDistributed_1_partial_volume_0']) * fitted_parameters['partial_volume_1'])

            # Calculate MSE and R2
            mse = NODDI_fit.mean_squared_error(data_for_bval)
            R2 = NODDI_fit.R2_coefficient_of_determination(data_for_bval)

            # Save results
            save_nifti(os.path.join(noddi_path, f'{patient_path}_b{bval}_mu.nii.gz'), mu.astype(np.float32), affine)
            save_nifti(os.path.join(noddi_path, f'{patient_path}_b{bval}_odi.nii.gz'), odi.astype(np.float32), affine)
            save_nifti(os.path.join(noddi_path, f'{patient_path}_b{bval}_fiso.nii.gz'), f_iso.astype(np.float32), affine)
            save_nifti(os.path.join(noddi_path, f'{patient_path}_b{bval}_fbundle.nii.gz'), f_bundle.astype(np.float32), affine)
            save_nifti(os.path.join(noddi_path, f'{patient_path}_b{bval}_fintra.nii.gz'), f_intra.astype(np.float32), affine)
            save_nifti(os.path.join(noddi_path, f'{patient_path}_b{bval}_icvf.nii.gz'), f_icvf.astype(np.float32), affine)
            save_nifti(os.path.join(noddi_path, f'{patient_path}_b{bval}_fextra.nii.gz'), f_extra.astype(np.float32), affine)
            save_nifti(os.path.join(noddi_path, f'{patient_path}_b{bval}_mse.nii.gz'), mse.astype(np.float32), affine)
            save_nifti(os.path.join(noddi_path, f'{patient_path}_b{bval}_R2.nii.gz'), R2.astype(np.float32), affine)

            print(f"[{log_prefix}] {datetime.datetime.now().strftime('%d.%b %Y %H:%M:%S')}: "
                  f"NODDI processing completed for b-value {bval}\n", flush=True)

    print(f"[{log_prefix}] {datetime.datetime.now().strftime('%d.%b %Y %H:%M:%S')}: "
          f"NODDI processing completed for patient {patient_path}\n", flush=True)



def process_noddi_data_per_pair(patient_path, preproc="preproc",b0_threshold=60, use_parallel_processing=True, number_of_processors=None):
    """
    Process NODDI data for a given patient for each unique b-value separately.

    Parameters:
    - patient_path (str): The name of the patient.
    - preproc (str): Type of preprocessing. Default is "preproc".
    - use_parallel_processing (bool): Whether to use parallel processing. Default is True.
    - number_of_processors (int): Number of processors to use for parallel processing. Default is None.

    Returns:
    - None
    """
    log_prefix = "NODDI Processing Per B-Value"
    print(f"[{log_prefix}] {datetime.datetime.now().strftime('%d.%b %Y %H:%M:%S')}: "
          f"Beginning processing for patient {patient_path}\n", flush=True)

    # Set up paths
    folder_path = os.path.dirname(os.getcwd())
    print(f"[{log_prefix}] Setting up paths...")
    noddi_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/noddi/"
    print(f"[{log_prefix}] NODDI path: {noddi_path}")

    fdwi = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/preproc/{patient_path}_dmri_preproc.nii.gz"
    fbval = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/preproc/{patient_path}_dmri_preproc.bval"
    fbvec = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/preproc/{patient_path}_dmri_preproc.bvec"
    
    print(f"[{log_prefix}] DWI: {fdwi}")
    print(f"[{log_prefix}] BVAL: {fbval}")
    print(f"[{log_prefix}] BVEC: {fbvec}")

    # Load data
    data, affine = load_nifti(fdwi)
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)

    # Identify unique b-values
    unique_bvals = np.unique(bvals)
    specific_bvals =[1000, 3000, 5000, 10000]
    bval_combinations = [(b1, b2) for i, b1 in enumerate(specific_bvals) for b2 in specific_bvals[i+1:] if b1 != b2]

    for bval_pair in bval_combinations:
        print(bval_pair)
        bval_indices = np.where((abs(bvals - bval_pair[0]) <= 150) | 
                        (abs(bvals - bval_pair[1]) <= 150) | 
                        (bvals == 0))[0]

        if len(bval_indices) > 0:
            # Select data for the current b-value
            data_for_bval = data[..., bval_indices]
            bvals_for_bval = bvals[bval_indices]
            bvecs_for_bval = bvecs[bval_indices]

            # Define NODDI models
            ball = gaussian_models.G1Ball()
            stick = cylinder_models.C1Stick()
            zeppelin = gaussian_models.G2Zeppelin()

            watson_dispersed_bundle = SD1WatsonDistributed(models=[stick, zeppelin])
            watson_dispersed_bundle.set_tortuous_parameter('G2Zeppelin_1_lambda_perp', 'C1Stick_1_lambda_par', 'partial_volume_0')
            watson_dispersed_bundle.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
            watson_dispersed_bundle.set_fixed_parameter('G2Zeppelin_1_lambda_par', 1.7e-9)

            NODDI_mod = MultiCompartmentModel(models=[ball, watson_dispersed_bundle])
            NODDI_mod.set_fixed_parameter('G1Ball_1_lambda_iso', 3.e-9)

            # Load mask
            mask,_ = load_nifti(f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/masks/{patient_path}_brain_mask_dilated.nii.gz")

            # Transform bval, bvecs
            b0_threshold = np.min(bvals_for_bval) + 10
            b0_threshold = max(50, b0_threshold)
            gtab_dipy = gradient_table(bvals_for_bval, bvecs_for_bval, b0_threshold=b0_threshold)
            acq_scheme_dmipy = gtab_dipy2dmipy(gtab_dipy, b0_threshold=b0_threshold*1e6)

            # Fit NODDI model
            NODDI_fit = NODDI_mod.fit(acq_scheme_dmipy, data_for_bval, mask=mask,
                                       use_parallel_processing=use_parallel_processing,
                                       number_of_processors=number_of_processors)

            # Get fitted parameters
            fitted_parameters = NODDI_fit.fitted_parameters
            mu = fitted_parameters["SD1WatsonDistributed_1_SD1Watson_1_mu"]
            odi = fitted_parameters["SD1WatsonDistributed_1_SD1Watson_1_odi"]
            f_iso = fitted_parameters["partial_volume_0"]
            f_bundle = fitted_parameters["partial_volume_1"]
            f_intra = (fitted_parameters['SD1WatsonDistributed_1_partial_volume_0'] * fitted_parameters['partial_volume_1'])
            f_icvf = fitted_parameters['SD1WatsonDistributed_1_partial_volume_0'] * (fitted_parameters['partial_volume_1'] > 0.05)
            f_extra = ((1 - fitted_parameters['SD1WatsonDistributed_1_partial_volume_0']) * fitted_parameters['partial_volume_1'])

            # Calculate MSE and R2
            mse = NODDI_fit.mean_squared_error(data_for_bval)
            R2 = NODDI_fit.R2_coefficient_of_determination(data_for_bval)

            bval_str = f"b{bval_pair[0]}b{bval_pair[1]}"

            # Save results
            save_nifti(os.path.join(noddi_path, f'{patient_path}_{bval_str}_mu.nii.gz'), mu.astype(np.float32), affine)
            save_nifti(os.path.join(noddi_path, f'{patient_path}_{bval_str}_odi.nii.gz'), odi.astype(np.float32), affine)
            save_nifti(os.path.join(noddi_path, f'{patient_path}_{bval_str}_fiso.nii.gz'), f_iso.astype(np.float32), affine)
            save_nifti(os.path.join(noddi_path, f'{patient_path}_{bval_str}_fbundle.nii.gz'), f_bundle.astype(np.float32), affine)
            save_nifti(os.path.join(noddi_path, f'{patient_path}_{bval_str}_fintra.nii.gz'), f_intra.astype(np.float32), affine)
            save_nifti(os.path.join(noddi_path, f'{patient_path}_{bval_str}_icvf.nii.gz'), f_icvf.astype(np.float32), affine)
            save_nifti(os.path.join(noddi_path, f'{patient_path}_{bval_str}_fextra.nii.gz'), f_extra.astype(np.float32), affine)
            save_nifti(os.path.join(noddi_path, f'{patient_path}_{bval_str}_mse.nii.gz'), mse.astype(np.float32), affine)
            save_nifti(os.path.join(noddi_path, f'{patient_path}_{bval_str}_R2.nii.gz'), R2.astype(np.float32), affine)

            print(f"[{log_prefix}] {datetime.datetime.now().strftime('%d.%b %Y %H:%M:%S')}: "
                  f"NODDI processing completed for {bval_str}\n", flush=True)

    print(f"[{log_prefix}] {datetime.datetime.now().strftime('%d.%b %Y %H:%M:%S')}: "
          f"NODDI processing completed for patient {patient_path}\n", flush=True)
    

def process_noddi_data_per_triple(patient_path, preproc="preproc",b0_threshold=60, use_parallel_processing=True, number_of_processors=None):
    """
    Process NODDI data for a given patient for each unique b-value separately.

    Parameters:
    - patient_path (str): The name of the patient.
    - preproc (str): Type of preprocessing. Default is "preproc".
    - use_parallel_processing (bool): Whether to use parallel processing. Default is True.
    - number_of_processors (int): Number of processors to use for parallel processing. Default is None.

    Returns:
    - None
    """
    log_prefix = "NODDI Processing Per B-Value"
    print(f"[{log_prefix}] {datetime.datetime.now().strftime('%d.%b %Y %H:%M:%S')}: "
          f"Beginning processing for patient {patient_path}\n", flush=True)

    # Set up paths
    folder_path = os.path.dirname(os.getcwd())
    print(f"[{log_prefix}] Setting up paths...")
    noddi_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/noddi/"
    print(f"[{log_prefix}] NODDI path: {noddi_path}")

    fdwi = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/preproc/{patient_path}_dmri_preproc.nii.gz"
    fbval = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/preproc/{patient_path}_dmri_preproc.bval"
    fbvec = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/preproc/{patient_path}_dmri_preproc.bvec"
    
    print(f"[{log_prefix}] DWI: {fdwi}")
    print(f"[{log_prefix}] BVAL: {fbval}")
    print(f"[{log_prefix}] BVEC: {fbvec}")

    # Load data
    data, affine = load_nifti(fdwi)
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)

    # Identify unique b-values
    unique_bvals = np.unique(bvals)
    specific_bvals = [1000, 3000, 5000, 10000]

    # Generating all unique combinations of 3 values from the list
    bval_combinations = list(combinations(specific_bvals, 3))

    # Printing the combinations
    for combination in bval_combinations:
        print(combination)

    for bval_pair in bval_combinations:
        print(bval_pair)
        bval_indices = np.where((abs(bvals - bval_pair[0]) <= 150) | 
                        (abs(bvals - bval_pair[1]) <= 150) | (abs(bvals - bval_pair[2]) <= 150) |
                        (bvals == 0))[0]

        if len(bval_indices) > 0:
            # Select data for the current b-value
            data_for_bval = data[..., bval_indices]
            bvals_for_bval = bvals[bval_indices]
            bvecs_for_bval = bvecs[bval_indices]

            # Define NODDI models
            ball = gaussian_models.G1Ball()
            stick = cylinder_models.C1Stick()
            zeppelin = gaussian_models.G2Zeppelin()

            watson_dispersed_bundle = SD1WatsonDistributed(models=[stick, zeppelin])
            watson_dispersed_bundle.set_tortuous_parameter('G2Zeppelin_1_lambda_perp', 'C1Stick_1_lambda_par', 'partial_volume_0')
            watson_dispersed_bundle.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
            watson_dispersed_bundle.set_fixed_parameter('G2Zeppelin_1_lambda_par', 1.7e-9)

            NODDI_mod = MultiCompartmentModel(models=[ball, watson_dispersed_bundle])
            NODDI_mod.set_fixed_parameter('G1Ball_1_lambda_iso', 3.e-9)

            # Load mask
            mask,_ = load_nifti(f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/masks/{patient_path}_brain_mask_dilated.nii.gz")

            # Transform bval, bvecs
            b0_threshold = np.min(bvals_for_bval) + 10
            b0_threshold = max(50, b0_threshold)
            gtab_dipy = gradient_table(bvals_for_bval, bvecs_for_bval, b0_threshold=b0_threshold)
            acq_scheme_dmipy = gtab_dipy2dmipy(gtab_dipy, b0_threshold=b0_threshold*1e6)

            # Fit NODDI model
            NODDI_fit = NODDI_mod.fit(acq_scheme_dmipy, data_for_bval, mask=mask,
                                       use_parallel_processing=use_parallel_processing,
                                       number_of_processors=number_of_processors)

            # Get fitted parameters
            fitted_parameters = NODDI_fit.fitted_parameters
            mu = fitted_parameters["SD1WatsonDistributed_1_SD1Watson_1_mu"]
            odi = fitted_parameters["SD1WatsonDistributed_1_SD1Watson_1_odi"]
            f_iso = fitted_parameters["partial_volume_0"]
            f_bundle = fitted_parameters["partial_volume_1"]
            f_intra = (fitted_parameters['SD1WatsonDistributed_1_partial_volume_0'] * fitted_parameters['partial_volume_1'])
            f_icvf = fitted_parameters['SD1WatsonDistributed_1_partial_volume_0'] * (fitted_parameters['partial_volume_1'] > 0.05)
            f_extra = ((1 - fitted_parameters['SD1WatsonDistributed_1_partial_volume_0']) * fitted_parameters['partial_volume_1'])

            # Calculate MSE and R2
            mse = NODDI_fit.mean_squared_error(data_for_bval)
            R2 = NODDI_fit.R2_coefficient_of_determination(data_for_bval)

            bval_str = f"b{bval_pair[0]}b{bval_pair[1]}b{bval_pair[2]}"

            # Save results
            save_nifti(os.path.join(noddi_path, f'{patient_path}_{bval_str}_mu.nii.gz'), mu.astype(np.float32), affine)
            save_nifti(os.path.join(noddi_path, f'{patient_path}_{bval_str}_odi.nii.gz'), odi.astype(np.float32), affine)
            save_nifti(os.path.join(noddi_path, f'{patient_path}_{bval_str}_fiso.nii.gz'), f_iso.astype(np.float32), affine)
            save_nifti(os.path.join(noddi_path, f'{patient_path}_{bval_str}_fbundle.nii.gz'), f_bundle.astype(np.float32), affine)
            save_nifti(os.path.join(noddi_path, f'{patient_path}_{bval_str}_fintra.nii.gz'), f_intra.astype(np.float32), affine)
            save_nifti(os.path.join(noddi_path, f'{patient_path}_{bval_str}_icvf.nii.gz'), f_icvf.astype(np.float32), affine)
            save_nifti(os.path.join(noddi_path, f'{patient_path}_{bval_str}_fextra.nii.gz'), f_extra.astype(np.float32), affine)
            save_nifti(os.path.join(noddi_path, f'{patient_path}_{bval_str}_mse.nii.gz'), mse.astype(np.float32), affine)
            save_nifti(os.path.join(noddi_path, f'{patient_path}_{bval_str}_R2.nii.gz'), R2.astype(np.float32), affine)

            print(f"[{log_prefix}] {datetime.datetime.now().strftime('%d.%b %Y %H:%M:%S')}: "
                  f"NODDI processing completed for {bval_str}\n", flush=True)

    print(f"[{log_prefix}] {datetime.datetime.now().strftime('%d.%b %Y %H:%M:%S')}: "
          f"NODDI processing completed for patient {patient_path}\n", flush=True)
#####################################################################################################################################
# Loop over patient IDs from sub-1001 to sub-1010
for i in range(1002, 1003):
    patient_path = f"sub-{i:04d}"
    use_parallel_processing = True
    number_of_processors = 25
    
    # Call process_noddi_data function for each patient
    #process_noddi_data(patient_path, preproc="preproc", b0_threshold=50, use_parallel_processing=True,
    #                   number_of_processors=None)

    #process_noddi_data_per_bval(patient_path, preproc="preproc",b0_threshold=60,  
    #                           use_parallel_processing=use_parallel_processing, number_of_processors=number_of_processors)
    
    #process_noddi_data_per_pair(patient_path, preproc="preproc",b0_threshold=60, use_parallel_processing=True, number_of_processors=number_of_processors)
    
    process_noddi_data_per_triple(patient_path, preproc="preproc",b0_threshold=60, use_parallel_processing=True, number_of_processors=number_of_processors)
