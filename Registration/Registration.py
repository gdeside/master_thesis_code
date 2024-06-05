# Import necessary libraries for neuroimaging data processing and visualization
from dipy.core.gradients import gradient_table
from dipy.io import read_bvals_bvecs
from nilearn import datasets, image, plotting
import numpy as np
import nibabel as nib
from dipy.viz import regtools
from dipy.segment.mask import applymask
from dipy.io.image import load_nifti, save_nifti
from dipy.align.imaffine import (transform_centers_of_mass, AffineMap, MutualInformationMetric, AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D, RigidTransform3D, AffineTransform3D)
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric
import pickle
import os
from dipy.data import get_sphere
import dipy.reconst.shm as shm
import dipy.direction.peaks as dp
import copy

def regToT1fromB0(reg_path, T1_subject, DWI_subject, mask_file, metrics_dic, folderpath, p, mapping_T1_to_T1MNI, T1_MNI,
                  mask_static, FA_MNI):
    print("Start of regToT1fromB0 function")
    if os.path.exists(reg_path + 'mapping_DWI_B0_to_T1.p'):
        with open(reg_path + 'mapping_DWI_B0_to_T1.p', 'rb') as handle:
            mapping_DWI_to_T1 = pickle.load(handle)
    else:
        if not (os.path.exists(reg_path)):
            try:
                os.makedirs(reg_path)
            except OSError:
                print("Creation of the directory %s failed" % reg_path)
        mapping_DWI_to_T1 = getTransform(T1_subject, DWI_subject, mask_file=mask_file, onlyAffine=False,
                                         diffeomorph=False, sanity_check=False, DWI=True)
        with open(reg_path + 'mapping_DWI_B0_to_T1.p', 'wb') as handle:
            pickle.dump(mapping_DWI_to_T1, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if not (os.path.exists(folderpath + "/subjects/" + p + "/masks/reg/")):
        try:
            os.makedirs(folderpath + "/subjects/" + p + "/masks/reg/")
        except OSError:
            print("Creation of the directory %s failed" % folderpath + "/subjects/" + p + "/masks/reg/")

    for maskType in ["brain_mask_dilated", "brain_mask", "wm_mask_MSMT", "wm_mask_AP", "wm_mask_FSL_T1",
                     "wm_mask_Freesurfer_T1"]:
        in_mask_path = folderpath + "/subjects/" + p + "/masks/" + p + "_" + maskType + ".nii.gz"
        reg_mask_path = folderpath + "/subjects/" + p + "/masks/reg/" + p + "_B0_" + maskType + ".nii.gz"
        if os.path.exists(in_mask_path):
            print("Applying transformation to mask:", in_mask_path)
            applyTransform(in_mask_path, mapping_DWI_to_T1, mapping_2=mapping_T1_to_T1MNI, static_file=T1_MNI,
                           output_path=reg_mask_path, mask_file=None, binary=False, inverse=False,
                           mask_static=mask_static, static_fa_file=FA_MNI)

    for key, value in metrics_dic.items():
        input_folder = folderpath + '/subjects/' + p + '/dMRI/microstructure/' + value + '/'
        output_folder = folderpath + '/subjects/' + p + '/dMRI/microstructure/' + value + '_CommonSpace_T1_B0/'

        if not (os.path.exists(output_folder)):
            try:
                os.makedirs(output_folder)
            except OSError:
                print("Creation of the directory %s failed" % output_folder)

        print("Start of applyTransformToAllMapsInFolder for metrics ", value, ":", key)
        applyTransformToAllMapsInFolder(input_folder, output_folder, mapping_DWI_to_T1, mapping_2=mapping_T1_to_T1MNI,
                                         static_file=T1_MNI, mask_file=mask_file, keywordList=[p, key], inverse=False,
                                         mask_static=mask_static, static_fa_file=FA_MNI)



def regToT1fromWMFOD(reg_path, T1_subject, WM_FOD_subject, mask_file, metrics_dic, folderpath, p, mapping_T1_to_T1MNI,
                     T1_MNI, mask_static, FA_MNI):
    print("Start of regToT1fromWMFOD function")
    if os.path.exists(reg_path + 'mapping_DWI_WMFOD_to_T1.p'):
        with open(reg_path + 'mapping_DWI_WMFOD_to_T1.p', 'rb') as handle:
            mapping_DWI_to_T1 = pickle.load(handle)
    else:
        if not (os.path.exists(reg_path)):
            try:
                os.makedirs(reg_path)
            except OSError:
                print("Creation of the directory %s failed" % reg_path)
        mapping_DWI_to_T1 = getTransform(T1_subject, WM_FOD_subject, mask_file=mask_file, onlyAffine=False,
                                         diffeomorph=False, sanity_check=False, DWI=True)
        with open(reg_path + 'mapping_DWI_WMFOD_to_T1.p', 'wb') as handle:
            pickle.dump(mapping_DWI_to_T1, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if not (os.path.exists(folderpath + "/subjects/" + p + "/masks/reg/")):
        try:
            os.makedirs(folderpath + "/subjects/" + p + "/masks/reg/")
        except OSError:
            print("Creation of the directory %s failed" % folderpath + "/subjects/" + p + "/masks/reg/")

    for maskType in ["brain_mask_dilated", "brain_mask", "wm_mask_MSMT", "wm_mask_AP", "wm_mask_FSL_T1",
                     "wm_mask_Freesurfer_T1"]:
        in_mask_path = folderpath + "/subjects/" + p + "/masks/" + p + "_" + maskType + ".nii.gz"
        reg_mask_path = folderpath + "/subjects/" + p + "/masks/reg/" + p + "_WMFOD_" + maskType + ".nii.gz"
        if os.path.exists(in_mask_path):
            print("Applying transformation to mask:", in_mask_path)
            applyTransform(in_mask_path, mapping_DWI_to_T1, mapping_2=mapping_T1_to_T1MNI, static_file=T1_MNI,
                           output_path=reg_mask_path, mask_file=None, binary=False, inverse=False,
                           mask_static=mask_static, static_fa_file=FA_MNI)

    for key, value in metrics_dic.items():

        input_folder = folderpath + '/subjects/' + p + '/dMRI/microstructure/' + value + '/'
        output_folder = folderpath + '/subjects/' + p + '/dMRI/microstructure/' + value + '_CommonSpace_T1_WMFOD/'

        if not (os.path.exists(output_folder)):
            try:
                os.makedirs(output_folder)
            except OSError:
                print("Creation of the directory %s failed" % output_folder)

        print("Start of applyTransformToAllMapsInFolder for metrics ", value, ":", key)
        applyTransformToAllMapsInFolder(input_folder, output_folder, mapping_DWI_to_T1, mapping_2=mapping_T1_to_T1MNI,
                                         static_file=T1_MNI, mask_file=mask_file, keywordList=[p, key], inverse=False,
                                         mask_static=mask_static, static_fa_file=FA_MNI)




def regToT1fromAP(reg_path, T1_subject, AP_subject, mask_file, metrics_dic, folderpath, p, mapping_T1_to_T1MNI, T1_MNI,
                  mask_static, FA_MNI):
    print("Start of regToT1fromAP function")
    if os.path.exists(reg_path + 'mapping_DWI_AP_to_T1.p'):
        with open(reg_path + 'mapping_DWI_AP_to_T1.p', 'rb') as handle:
            mapping_DWI_to_T1 = pickle.load(handle)
    else:
        if not (os.path.exists(reg_path)):
            try:
                os.makedirs(reg_path)
            except OSError:
                print("Creation of the directory %s failed" % reg_path)
        print("Obtaining transform for DWI to T1 registration...")
        mapping_DWI_to_T1 = getTransform(T1_subject, AP_subject, mask_file=mask_file, onlyAffine=False,
                                         diffeomorph=False, sanity_check=False, DWI=False)
        with open(reg_path + 'mapping_DWI_AP_to_T1.p', 'wb') as handle:
            pickle.dump(mapping_DWI_to_T1, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if not (os.path.exists(folderpath + "/subjects/" + p + "/masks/reg/")):
        try:
            os.makedirs(folderpath + "/subjects/" + p + "/masks/reg/")
        except OSError:
            print("Creation of the directory %s failed" % folderpath + "/subjects/" + p + "/masks/reg/")

    for maskType in ["brain_mask_dilated","brain_mask", "wm_mask_MSMT", "wm_mask_AP", "wm_mask_FSL_T1",
                    "wm_mask_Freesurfer_T1"]:
        in_mask_path = folderpath + "/subjects/" + p + "/masks/" + p + "_" + maskType + ".nii.gz"
        reg_mask_path = folderpath + "/subjects/" + p + "/masks/reg/" + p + "_AP_" + maskType + ".nii.gz"
        if os.path.exists(in_mask_path):
            print("Applying transformation to mask:", in_mask_path)
            applyTransform(in_mask_path, mapping_DWI_to_T1, mapping_2=mapping_T1_to_T1MNI, static_file=T1_MNI,
                           output_path=reg_mask_path, mask_file=None, binary=False, inverse=False,
                           mask_static=mask_static, static_fa_file=FA_MNI)

    for key, value in metrics_dic.items():

        input_folder = folderpath + '/subjects/' + p + '/dMRI/microstructure/' + value + '/'
        output_folder = folderpath + '/subjects/' + p + '/dMRI/microstructure/' + value + '_CommonSpace_T1_AP/'

        if not (os.path.exists(output_folder)):
            try:
                os.makedirs(output_folder)
            except OSError:
                print("Creation of the directory %s failed" % output_folder)

        print("Start of applyTransformToAllMapsInFolder for metrics ", value, ":", key)
        applyTransformToAllMapsInFolder(input_folder, output_folder, mapping_DWI_to_T1, mapping_2=mapping_T1_to_T1MNI,
                                        static_file=T1_MNI, mask_file=mask_file, keywordList=[p, key], inverse=False,
                                        mask_static=mask_static, static_fa_file=FA_MNI)



def regallDWIToT1wToT1wCommonSpace(folder_path, p, DWI_type="AP", maskType=None, T1_filepath=None,
                                    T1wCommonSpace_filepath="${FSLDIR}/data/standard/MNI152_T1_1mm_brain.nii.gz",
                                      T1wCommonSpaceMask_filepath="${FSLDIR}/data/standard/MNI152_T1_1mm_brain_mask.nii.gz", 
                                      metrics_dic={'_FA': 'dti', 'RD': 'dti', 'AD': 'dti', 'MD': 'dti'}):
    print("Starting regallDWIToT1wToT1wCommonSpace function...")
    print("Arguments:")
    print("folder_path:", folder_path)
    print("p:", p)
    print("DWI_type:", DWI_type)
    print("maskType:", maskType)
    print("T1_filepath:", T1_filepath)
    print("T1wCommonSpace_filepath:", T1wCommonSpace_filepath)
    print("T1wCommonSpaceMask_filepath:", T1wCommonSpaceMask_filepath)
    print("metrics_dic:", metrics_dic)

    preproc_folder = folder_path + '/subjects/' + p + '/dMRI/preproc/'
    T1_CommonSpace = os.path.expandvars(T1wCommonSpace_filepath)
    FA_MNI = os.path.expandvars('${FSLDIR}/data/standard/FSL_HCP1065_FA_1mm.nii.gz')

    assert maskType in ["brain_mask_dilated","brain_mask", "wm_mask_MSMT", "wm_mask_AP", "wm_mask_FSL_T1",
                    "wm_mask_Freesurfer_T1", None], "The mask parameter must be one of the following : brain_mask_dilated, brain_mask, wm_mask_MSMT, wm_mask_AP, wm_mask_FSL_T1, wm_mask_Freesurfer_T1, None"

    assert DWI_type in ["AP", "WMFOD", "BO"], "The DWI_type parameter must be one of the following : AP, WMFOD, BO"

    mask_path = ""
    if maskType is not None and os.path.isfile(folder_path + '/subjects/' + p + "/masks/" + p + '_' + maskType + '.nii.gz'):
        mask_path = folder_path + '/subjects/' + p + "/masks/" + p + '_' + maskType + '.nii.gz'
    else:
        mask_path = None

    if T1_filepath is None:
        T1_subject = folder_path + '/subjects/' + p + '/T1/' + p + "_T1_brain.nii.gz"
    elif os.path.exists(os.path.join(T1_filepath,p + ".nii.gz")):
        T1_subject = os.path.join(T1_filepath,p + ".nii.gz")
    elif os.path.exists(os.path.join(T1_filepath,p + "_T1.nii.gz")):
        T1_subject = os.path.join(T1_filepath,p + "_T1.nii.gz")
    elif os.path.exists(os.path.join(T1_filepath,p + "_T1_brain.nii.gz")):
        T1_subject = os.path.join(T1_filepath,p + "_T1_brain.nii.gz")
    else:
        raise ValueError("No T1 file found in the T1_filepath folder")

    DWI_subject = preproc_folder + p + "_dmri_preproc.nii.gz"
    AP_subject = folder_path + '/subjects/' + p + '/masks/' + p + '_ap.nii.gz'
    WM_FOD_subject = folder_path + '/subjects/' + p + '/dMRI/ODF/MSMT-CSD/' + p + "_MSMT-CSD_WM_ODF.nii.gz"

    reg_path = folder_path + '/subjects/' + p + '/reg/'
    if not(os.path.exists(reg_path)):
        try:
            os.makedirs(reg_path)
        except OSError:
            print ("Creation of the directory %s failed" % reg_path)

    print("Start of getTransform for T1 to T1_MNI")
    mask_file = None

    if os.path.exists(reg_path + 'mapping_T1w_to_T1wCommonSpace.p'):
        with open(reg_path + 'mapping_T1w_to_T1wCommonSpace.p', 'rb') as handle:
            mapping_T1w_to_T1wCommonSpace = pickle.load(handle)
    else:
        if not (os.path.exists(reg_path)):
            try:
                os.makedirs(reg_path)
            except OSError:
                print("Creation of the directory %s failed" % reg_path)
        mapping_T1w_to_T1wCommonSpace = getTransform(T1_CommonSpace, T1_subject, mask_file=mask_file, onlyAffine=False, diffeomorph=True,
                                           sanity_check=False, DWI=False)
        with open(reg_path + 'mapping_T1w_to_T1wCommonSpace.p', 'wb') as handle:
            pickle.dump(mapping_T1w_to_T1wCommonSpace, handle, protocol=pickle.HIGHEST_PROTOCOL)

    applyTransform(T1_subject, mapping_T1w_to_T1wCommonSpace, mapping_2=None, mask_file=None, static_file=T1_CommonSpace,
                   output_path=folder_path + '/subjects/' + p + '/T1/' + p + '_T1_MNI_FS.nii.gz', binary=False,
                   inverse=False, static_fa_file=T1_CommonSpace)

    print("Start of getTransform for DWI to T1")
    if T1wCommonSpaceMask_filepath is not None:
        mask_static = os.path.expandvars(T1wCommonSpaceMask_filepath)
    else:
        mask_static = None

    if DWI_type == "B0":
        regToT1fromB0(reg_path, T1_subject, DWI_subject, mask_path, metrics_dic, folder_path, p, mapping_T1w_to_T1wCommonSpace, T1_CommonSpace, mask_static, FA_MNI)
    elif DWI_type == "WMFOD":
        regToT1fromWMFOD(reg_path, T1_subject, WM_FOD_subject, mask_path, metrics_dic, folder_path, p, mapping_T1w_to_T1wCommonSpace, T1_CommonSpace, mask_static, FA_MNI)
    elif DWI_type == "AP":
        regToT1fromAP(reg_path, T1_subject, AP_subject, mask_path, metrics_dic, folder_path, p, mapping_T1w_to_T1wCommonSpace, T1_CommonSpace, mask_static, FA_MNI)
    else:
        print("DWI_type not recognized")

    print("End of DWI registration")




def applyTransform(file_path, mapping, mapping_2=None, mask_file=None, static_file='', output_path='', binary=False,
                   inverse=False, mask_static=None, static_fa_file=''):
    print("Starting applyTransform function...")
    print("Arguments:")
    print("file_path:", file_path)
    print("mapping:", mapping)
    print("mapping_2:", mapping_2)
    print("mask_file:", mask_file)
    print("static_file:", static_file)
    print("output_path:", output_path)
    print("binary:", binary)
    print("inverse:", inverse)
    print("mask_static:", mask_static)
    print("static_fa_file:", static_fa_file)

    # Load the moving image data from the specified file path
    print("Loading moving image data...")
    moving = nib.load(file_path)
    moving_data = moving.get_fdata()

    # Apply a mask to the moving image data, if specified
    if mask_file is not None:
        print("Applying mask to moving image data...")
        mask, mask_affine = load_nifti(mask_file)
        moving_data = applymask(moving_data, mask)

    # Apply the primary transformation, or its inverse if specified
    print("Applying primary transformation...")
    if inverse:
        transformed = mapping.transform_inverse(moving_data)
    else:
        transformed = mapping.transform(moving_data)

    # Apply the secondary transformation, if specified
    if mapping_2 is not None:
        print("Applying secondary transformation...")
        if inverse:
            transformed = mapping_2.transform_inverse(transformed)
        else:
            transformed = mapping_2.transform(transformed)

    # Convert the image to binary if specified
    if binary:
        print("Converting image to binary...")
        transformed[transformed > .5] = 1
        transformed[transformed <= .5] = 0

    # Save the transformed image to the specified output path, if provided
    if len(output_path) > 0:
        print("Saving transformed image...")
        # Load the static image to use its affine for saving
        static = nib.load(static_file)

        # Load the file from which to copy header information, if specified
        static_fa = nib.load(static_fa_file)

        # Apply a mask to the static image, if specified
        if mask_static is not None:
            print("Applying mask to static image...")
            mask_static_data, mask_static_affine = load_nifti(mask_static)
            transformed = applymask(transformed, mask_static_data)

        # Create a new Nifti image with the transformed data, using the static image's affine and the copied header
        out = nib.Nifti1Image(transformed, static.affine, header=static_fa.header)
        # Save the new Nifti image to the specified output path
        out.to_filename(output_path)
        print("Transformation saved to:", output_path)
    else:
        # If no output path was specified, return the transformed image data
        print("Transformation complete.")
        return transformed


# Define the main function for performing the image transformation and registration
def getTransform(static_volume_file, moving_volume_file, mask_file=None, onlyAffine=False, diffeomorph=True, sanity_check=False, DWI=False):
    '''
    Perform image registration between static and moving volumes. Supports both affine and diffeomorphic transformations.

    Parameters
    ----------
    static_volume_file : str or nib.Nifti1Image
        The file path or Nifti image of the static (reference) volume.
    moving_volume_file : str or nib.Nifti1Image
        The file path or Nifti image of the moving (source) volume.
    mask_file : str, optional
        The file path of the mask image to be applied to the moving volume.
    onlyAffine : bool, default False
        If True, perform only affine registration without diffeomorphic refinement.
    diffeomorph : bool, default True
        If True, perform diffeomorphic registration after affine alignment.
    sanity_check : bool, default False
        If True, generate and save overlay images for registration quality assessment.
    DWI : bool, default False
        If True, handle the moving volume as a DWI image by extracting the first volume.

    Returns
    -------
    mapping : object
        The transformation mapping from moving to static volume space.
    '''

    # Load the moving volume
    static_data, static_affine = load_nifti(static_volume_file)
    static_grid2world = static_affine

    moving_data, moving_affine = load_nifti(moving_volume_file)
    moving_grid2world = moving_affine

    if DWI:
        moving = np.squeeze(moving)[..., 0]

    # Apply mask to moving volume if provided
    if mask_file is not None:
        mask, mask_affine = load_nifti(mask_file)
        print(f"Shape of the mask for {mask_file}: {mask.shape}")
        original_shape = moving_data.shape
        print(f"Original shape of moving data: {original_shape}")
        moving_data = applymask(moving_data, mask)
        print(f"Shape of moving data after applying mask: {moving_data.shape}")

    # Initialize identity matrix for affine registration
    # Perform sanity check by overlaying static and moving images if requested
    if sanity_check or onlyAffine:
        identity = np.eye(4)
        affine_map = AffineMap(identity, static_data.shape, static_grid2world, moving_data.shape, moving_grid2world)
        if sanity_check:
            resampled = affine_map.transform(moving_data)
            # Overlay slices for visual inspection at different axes
            regtools.overlay_slices(static_data, resampled, None, 0, "Static", "Moving", "resampled_0.png")
            regtools.overlay_slices(static_data, resampled, None, 1, "Static", "Moving", "resampled_1.png")
            regtools.overlay_slices(static_data, resampled, None, 2, "Static", "Moving", "resampled_2.png")
        if onlyAffine:
            return affine_map  # Return early if only affine registration is requested

    # Affine registration setup and execution
    # Start with translation, then rigid, and finally affine transformations
    c_of_mass = transform_centers_of_mass(static_data, static_grid2world, moving_data, moving_grid2world)
    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)
    level_iters = [10000, 1000, 100]
    sigmas = [3.0, 1.0, 0.0]
    factors = [4, 2, 1]
    affreg = AffineRegistration(metric=metric, level_iters=level_iters, sigmas=sigmas, factors=factors)

    transform = TranslationTransform3D()
    params0 = None
    translation = affreg.optimize(static_data, moving_data, transform, params0, static_grid2world, moving_grid2world, starting_affine=c_of_mass.affine)

    transform = RigidTransform3D()
    rigid = affreg.optimize(static_data, moving_data, transform, params0, static_grid2world, moving_grid2world, starting_affine=translation.affine)

    transform = AffineTransform3D()
    affine = affreg.optimize(static_data, moving_data, transform, params0, static_grid2world, moving_grid2world, starting_affine=rigid.affine)

    # Diffeomorphic registration (non-linear) if requested
    # This step refines the registration by accounting for non-linear deformations
    if diffeomorph:
        metric = CCMetric(3)
        level_iters = [10, 10, 5]
        sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)
        mapping = sdr.optimize(static_data, moving_data, static_affine, moving_affine, affine.affine)
    else:
        mapping = affine  # Use affine result if no diffeomorphic registration is performed

    # If sanity check is enabled, overlay transformed moving image onto static image for evaluation
    if sanity_check:
        transformed = mapping.transform(moving_data)
        regtools.overlay_slices(static_data, transformed, None, 0, "Static", "Transformed", "transformed.png")
        regtools.overlay_slices(static_data, transformed, None, 1, "Static", "Transformed", "transformed.png")
        regtools.overlay_slices(static_data, transformed, None, 2, "Static", "Transformed", "transformed.png")

    return mapping  # Return the final transformation mapping

####################################################################################################################################################################

for subject_id in range(1001, 1011):
    subject = f"sub-{subject_id}"
    print(f"Setting up for subject: {subject}")

    subject_folder = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{subject}"
    print(f"Subject folder set to: {subject_folder}")

    T1_subject_path = f'{subject_folder}/T1/{subject}_T1_brain.nii.gz'
    print(f"T1 image path set to: {T1_subject_path}")

    T1_subject = nib.load(T1_subject_path)
    T1_subject_data = T1_subject.get_fdata()

    output_folder = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{subject}/reg"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # This will create all intermediate directories as well if they don't exist

    
    with open(os.path.join(output_folder, 'mapping_DWI_AP_to_T1.p'), 'rb') as handle:
            mapping_DWI_to_T1 = pickle.load(handle)

            T1_DWIspace = mapping_DWI_to_T1.transform_inverse(T1_subject_data, interpolation='nearest')

            T1_DWIspace_path = output_folder +f"/{subject}_T1_DWIspace.nii.gz"


            T1_DWIspaceHeader = copy.deepcopy(T1_subject.header)
            T1_DWIspaceHeader["dim"][1:4] = T1_subject_data.shape[0:3]
            out_T1_DWI = nib.Nifti1Image(T1_DWIspace, None, T1_DWIspaceHeader)
            out_T1_DWI.to_filename(T1_DWIspace_path)

            print("Files created:")

            print("out_path:", T1_DWIspace_path)

            


