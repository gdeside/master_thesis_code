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

def process_noddi_data_reduce(patient_path,bval_reduced,nb_vectors, preproc="preproc",b0_threshold=60, use_parallel_processing=True, number_of_processors=None):
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
    b0_threshold = np.min(updated_bvals) + 10
    b0_threshold = max(50, b0_threshold)
    gtab_dipy = gradient_table(updated_bvals, updated_bvecs, b0_threshold=b0_threshold)
    acq_scheme_dmipy = gtab_dipy2dmipy(gtab_dipy, b0_threshold=b0_threshold*1e6)
    
    # Fit NODDI model
    NODDI_fit = NODDI_mod.fit(acq_scheme_dmipy, updated_data, mask=mask,
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
    mse = NODDI_fit.mean_squared_error(updated_data)
    R2 = NODDI_fit.R2_coefficient_of_determination(updated_data)

    # Save results
    save_nifti(os.path.join(noddi_path, f'{patient_path}_reduced_b{bval_reduced}_{nb_vectors}_mu.nii.gz'), mu.astype(np.float32), affine)
    save_nifti(os.path.join(noddi_path, f'{patient_path}_reduced_b{bval_reduced}_{nb_vectors}_odi.nii.gz'), odi.astype(np.float32), affine)
    save_nifti(os.path.join(noddi_path, f'{patient_path}_reduced_b{bval_reduced}_{nb_vectors}_fiso.nii.gz'), f_iso.astype(np.float32), affine)
    save_nifti(os.path.join(noddi_path, f'{patient_path}_reduced_b{bval_reduced}_{nb_vectors}_fbundle.nii.gz'), f_bundle.astype(np.float32), affine)
    save_nifti(os.path.join(noddi_path, f'{patient_path}_reduced_b{bval_reduced}_{nb_vectors}_fintra.nii.gz'), f_intra.astype(np.float32), affine)
    save_nifti(os.path.join(noddi_path, f'{patient_path}_reduced_b{bval_reduced}_{nb_vectors}_icvf.nii.gz'), f_icvf.astype(np.float32), affine)
    save_nifti(os.path.join(noddi_path, f'{patient_path}_reduced_b{bval_reduced}_{nb_vectors}_fextra.nii.gz'), f_extra.astype(np.float32), affine)
    save_nifti(os.path.join(noddi_path, f'{patient_path}_reduced_b{bval_reduced}_{nb_vectors}_mse.nii.gz'), mse.astype(np.float32), affine)
    save_nifti(os.path.join(noddi_path, f'{patient_path}_reduced_b{bval_reduced}_{nb_vectors}_R2.nii.gz'), R2.astype(np.float32), affine)

    print(f"[{log_prefix}] {datetime.datetime.now().strftime('%d.%b %Y %H:%M:%S')}: "
          f"NODDI processing completed for patient {patient_path}\n", flush=True)
    

#####################################################################################################################################
# Loop over patient IDs from sub-1001 to sub-1010
for i in range(1009, 1011):
    patient_path = f"sub-{i:04d}"
    use_parallel_processing = True
    number_of_processors = 10
    
    # Call process_noddi_data function for each patient
    #process_noddi_data_reduce(patient_path,10000,128, preproc="preproc", b0_threshold=50, use_parallel_processing=True,
    #                   number_of_processors=number_of_processors)

    #process_noddi_data_reduce(patient_path,3000,48, preproc="preproc", b0_threshold=50, use_parallel_processing=True,
    #                   number_of_processors=number_of_processors)
    
    process_noddi_data_reduce(patient_path,10000,128, preproc="preproc", b0_threshold=50, use_parallel_processing=True,
                       number_of_processors=number_of_processors)
    
    #process_noddi_data_reduce(patient_path,10000,200, preproc="preproc", b0_threshold=50, use_parallel_processing=True,
    #                   number_of_processors=number_of_processors)

