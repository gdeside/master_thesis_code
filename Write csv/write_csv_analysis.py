from dipy.io.image import load_nifti
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

import csv

################################################################### MSE #############################################################

def calculate_mse_multi_methods_write_csv(patient_range, method_list):
    """
    Writes MSE (Mean Squared Error) values for multiple methods and patients into a CSV file. Each row in the CSV
    represents a patient, with columns for the patient ID and one column for each method to hold an MSE value.

    Args:
    - patient_range (range): Range of patient IDs to process.
    - method_list (list): List of MRI method names in the order ["dti", "noddi", "mf"].

    Returns:
    - None: Outputs a CSV file with MSE values.
    """
    folder_path = os.path.dirname(os.getcwd())
    writing_path = os.path.join(folder_path, "results/mse", "mse_values_comparison_method.csv")

    # Ensure the method list is in the expected order

    with open(writing_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Patient ID'] + method_list)
        
        for patient_id in patient_range:
            patient_path = f"sub-{patient_id}"
            mse_values = []
            
            for method_name in method_list:
                if method_name == "MF":
                    method_path = os.path.join("/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects", patient_path, "dMRI/microstructure/mf", f"{patient_path}_MSE.nii.gz")
                else:
                    method_path = os.path.join("/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects", patient_path, "dMRI", method_name, f"{patient_path}_mse.nii.gz")
                
                mse, _ = load_nifti(method_path)
                brain_mask_path = os.path.join("/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects", patient_path, "masks", f"{patient_path}_wm_mask_AP.nii.gz")
                brain_mask, _ = load_nifti(brain_mask_path)
                brain_mask = np.isin(brain_mask, [1], invert=True)
                
                mse = ma.masked_array(mse, mask=brain_mask).compressed()
                mse = mse[~np.isnan(mse)]

                mse_values.append(mse)

            nb_patient = len(mse_values)//3
            for i in range(nb_patient):
                nb_element = len(mse_values[i*3])
                for j in range(nb_element):
                    row = [patient_path]
                    row.append(mse_values[i*3][j])
                    row.append(mse_values[i*3 +1][j])
                    row.append(mse_values[i*3 +2][j])
                    writer.writerow(row)

    print(f"MSE values comparison written to {writing_path}")
    return 


def calculate_mse_multi_methods_write_csv_ROI(patient_range, method_list,atlas_name, atlas_values, roi_name):
    """
    Writes MSE (Mean Squared Error) values for multiple methods and patients into a CSV file. Each row in the CSV
    represents a patient, with columns for the patient ID and one column for each method to hold an MSE value.

    Args:
    - patient_range (range): Range of patient IDs to process.
    - method_list (list): List of MRI method names in the order ["dti", "noddi", "mf"].

    Returns:
    - None: Outputs a CSV file with MSE values.
    """
    folder_path = os.path.dirname(os.getcwd())

    results_path = os.path.join(folder_path, f"results/mse/{roi_name}")
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    writing_path = os.path.join(folder_path, f"results/mse/{roi_name}", f"mse_values_comparison_method_{roi_name}.csv")



    # Ensure the method list is in the expected order

    with open(writing_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Patient ID'] + method_list)
        
        for patient_id in patient_range:
            patient_path = f"sub-{patient_id}"
            subject_folder = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}"
            mse_values = []
            
            for method_name in method_list:
                if method_name == "MF":
                    method_path = os.path.join("/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects", patient_path, "dMRI/microstructure/mf", f"{patient_path}_MSE.nii.gz")
                else:
                    method_path = os.path.join("/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects", patient_path, "dMRI", method_name, f"{patient_path}_mse.nii.gz")
                
                mse, _ = load_nifti(method_path)
                brain_mask_path = f"{subject_folder}/reg/{patient_path}_Atlas_{atlas_name}_InSubjectDWISpaceFrom_AP.nii.gz"
                brain_mask, _ = load_nifti(brain_mask_path)
                brain_mask_mask = np.where(np.isin(brain_mask, atlas_values), 0, 1)
                
                mse = ma.masked_array(mse, mask=brain_mask_mask).compressed()
                mse = mse[~np.isnan(mse)]

                mse_values.append(mse)

            nb_patient = len(mse_values)//3
            for i in range(nb_patient):
                nb_element = len(mse_values[i*3])
                for j in range(nb_element):
                    row = [patient_path]
                    row.append(mse_values[i*3][j])
                    row.append(mse_values[i*3 +1][j])
                    row.append(mse_values[i*3 +2][j])
                    writer.writerow(row)

    print(f"MSE values comparison written to {writing_path}")
    return 


def calculate_method_MSE_by_bvalue_pair_write_csv(patient_range, method_name, prepoc_raw="preproc"):
    """
    Calculate and plot the boxplot of Mean Squared Error (MSE) for a specific method and multiple b-values.

    Parameters:
    - patient_path (str): The name of the patient.
    - method_name (str): Name of the method (e.g., "noddi", "mf", "dti").
    - show (bool): Whether to display the plot using plt.show(). Default is True.
    - save_as_png (bool): Whether to save the plot as a PNG file. Default is False.

    Returns:
    - None
    """
    folder_path = os.path.dirname(os.getcwd())
    writing_path = os.path.join(folder_path, "results/mse", f"mse_values_comparison_{method_name}_pair.csv")


    if method_name == "DTI":
        b_values = [1000, 3000]
    else:
        b_values = [1000, 3000, 5000, 10000]

    bval_combinations = [(b1, b2) for i, b1 in enumerate(b_values) for b2 in b_values[i+1:] if b1 != b2]

    # Ensure the method list is in the expected order

    with open(writing_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Patient ID'] + bval_combinations +["All shells"])
        
        for patient_id in patient_range:
            patient_path = f"sub-{patient_id}"
            mse_values = []

            if method_name == "MF":
                method_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/"
            else:
                method_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/{method_name}/"



            labels = []

            for bval_pair in bval_combinations:
                bval_str = f"b{bval_pair[0]}b{bval_pair[1]}"
                labels.append(bval_str)

                if method_name == "DTI":
                    mse, _ = load_nifti(method_path + f"{patient_path}_mse_{bval_str}.nii.gz")
                
                elif method_name == "MF":
                    print("MF")
                    mse, _ = load_nifti(method_path+f"{patient_path}_{bval_str}_MSE.nii.gz")
                else:
                    mse, _ = load_nifti(method_path + f"{patient_path}_{bval_str}_mse.nii.gz")


                brain_mask_path = os.path.join("/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects", patient_path, "masks", f"{patient_path}_wm_mask_AP.nii.gz")
                brain_mask, _ = load_nifti(brain_mask_path)
                brain_mask = np.isin(brain_mask, [1], invert=True)
                
                mse = ma.masked_array(mse, mask=brain_mask).compressed()
                mse = mse[~np.isnan(mse)]

                mse_values.append(mse)

            if method_name == "MF":
                method_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/"
            else:
                method_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/{method_name}/"

            if method_name == "DTI":
                 mse, _ = load_nifti(method_path + f"{patient_path}_mse.nii.gz")
                
            elif method_name == "MF":
                print("MF")
                mse, _ = load_nifti(method_path+f"{patient_path}_MSE.nii.gz")
            else:
                mse, _ = load_nifti(method_path + f"{patient_path}_mse.nii.gz")


            brain_mask_path = os.path.join("/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects", patient_path, "masks", f"{patient_path}_wm_mask_AP.nii.gz")
            brain_mask, _ = load_nifti(brain_mask_path)
            brain_mask = np.isin(brain_mask, [1], invert=True)
                
            mse = ma.masked_array(mse, mask=brain_mask).compressed()
            mse = mse[~np.isnan(mse)]

            mse_values.append(mse)

            nb_combination = len(bval_combinations) +1

            nb_patient = len(mse_values)//nb_combination
            print(nb_patient,len(bval_combinations))
            for i in range(nb_patient):
                nb_element = len(mse_values[i*nb_combination])
                for j in range(nb_element):
                    row = [patient_path]
                    for k in range(nb_combination):
                        row.append(mse_values[i*nb_combination+k][j])
                    writer.writerow(row)

    print(f"MSE values comparison written to {writing_path}")
    return 


def calculate_mse_method_write_csv_ROI(patient_range, method_name, atlas_name, atlas_dico):
    """
    Writes MSE (Mean Squared Error) values for a specified MRI method and patients into separate CSV files for each ROI.

    Args:
    - patient_range (range): Range of patient IDs to process.
    - method_name (str): Name of the MRI method (e.g., 'DTI').
    - atlas_name (str): Name of the atlas used for ROI segmentation.
    - atlas_dico (dict): Dictionary with atlas ROI IDs mapped to descriptive names.

    Returns:
    - None: Outputs a CSV file for each ROI with MSE values.
    """
    folder_path = os.path.dirname(os.getcwd())
    results_path = os.path.join(folder_path, "results/mse/")
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    for key, value in atlas_dico.items():
        writing_path = os.path.join(results_path, f"mse_{method_name}_{value.replace(' ', '_').lower()}.csv")
        
        with open(writing_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Patient ID', 'MSE'])
            
            for patient_id in patient_range:
                patient_path = f"sub-{patient_id}"
                subject_folder = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}"
                
                if method_name == "MF":
                    method_path = os.path.join(subject_folder, "dMRI/microstructure/mf", f"{patient_path}_MSE.nii.gz")
                else:
                    method_path = os.path.join(subject_folder, "dMRI", method_name, f"{patient_path}_mse.nii.gz")
                
                mse, _ = load_nifti(method_path)
                brain_mask_path = os.path.join(subject_folder, "reg", f"{patient_path}_Atlas_{atlas_name}_InSubjectDWISpaceFrom_AP.nii.gz")
                brain_mask, _ = load_nifti(brain_mask_path)
                brain_mask_mask = np.isin(brain_mask, [key], invert=True)
                
                mse = ma.masked_array(mse, mask=brain_mask_mask).compressed()
                mse = mse[~np.isnan(mse)]
                
                # Calculate statistics or simply write all MSE values
                for mse_value in mse:
                    writer.writerow([patient_path, mse_value])
        
        print(f"MSE values for {value} written to {writing_path}")


def calculate_method_MSE_reduced_write_csv(patient_range, method_name,bval, prepoc_raw="preproc"):
    """
    Calculate and plot the boxplot of Mean Squared Error (MSE) for a specific method and multiple b-values.

    Parameters:
    - patient_path (str): The name of the patient.
    - method_name (str): Name of the method (e.g., "noddi", "mf", "dti").
    - show (bool): Whether to display the plot using plt.show(). Default is True.
    - save_as_png (bool): Whether to save the plot as a PNG file. Default is False.

    Returns:
    - None
    """
    folder_path = os.path.dirname(os.getcwd())
    writing_path = os.path.join(folder_path, "results/mse", f"mse_values_comparison_{method_name}_reduced_{bval}.csv")


    if bval == 1000:
        lst_directions = [16,32,40,48,64]

    elif bval == 10000:
        lst_directions = [128,200,228]

    elif bval == 3000:
        lst_directions = [16,32,40,48,64]
    else:
        lst_directions = [32, 64, 100, 128]



    # Ensure the method list is in the expected order

    with open(writing_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Patient ID'] + lst_directions)
        
        for patient_id in patient_range:
            patient_path = f"sub-{patient_id}"
            mse_values = []

            if method_name == "MF":
                method_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/"
            else:
                method_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/{method_name}/"



            labels = []

            for direction in lst_directions[:-1]:

                if method_name == "DTI":
                    mse, _ = load_nifti(method_path + f"{patient_path}_reduced_b{bval}_{direction}_mse.nii.gz")
                
                elif method_name == "MF":
                    print("MF")
                    mse, _ = load_nifti(method_path+f"{patient_path}_reduced_b{bval}_{direction}_MSE.nii.gz")
                else:
                    mse, _ = load_nifti(method_path + f"{patient_path}_reduced_b{bval}_{direction}_mse.nii.gz")


                brain_mask_path = os.path.join("/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects", patient_path, "masks", f"{patient_path}_wm_mask_AP.nii.gz")
                brain_mask, _ = load_nifti(brain_mask_path)
                brain_mask = np.isin(brain_mask, [1], invert=True)
                
                mse = ma.masked_array(mse, mask=brain_mask).compressed()
                mse = mse[~np.isnan(mse)]

                mse_values.append(mse)

            if method_name == "MF":
                method_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/"
            else:
                method_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/{method_name}/"

            if method_name == "DTI":
                 mse, _ = load_nifti(method_path + f"{patient_path}_mse.nii.gz")
                
            elif method_name == "MF":
                print("MF")
                mse, _ = load_nifti(method_path+f"{patient_path}_MSE.nii.gz")
            else:
                mse, _ = load_nifti(method_path + f"{patient_path}_mse.nii.gz")


            brain_mask_path = os.path.join("/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects", patient_path, "masks", f"{patient_path}_wm_mask_AP.nii.gz")
            brain_mask, _ = load_nifti(brain_mask_path)
            brain_mask = np.isin(brain_mask, [1], invert=True)
                
            mse = ma.masked_array(mse, mask=brain_mask).compressed()
            mse = mse[~np.isnan(mse)]

            mse_values.append(mse)

            nb_combination = len(lst_directions) 

            nb_patient = len(mse_values)//nb_combination
            print(nb_patient,len(lst_directions))
            for i in range(nb_patient):
                nb_element = len(mse_values[i*nb_combination])
                for j in range(nb_element):
                    row = [patient_path]
                    for k in range(nb_combination):
                        row.append(mse_values[i*nb_combination+k][j])
                    writer.writerow(row)

    print(f"MSE values comparison written to {writing_path}")
    return 

def calculate_method_MSE_by_bvalue_write_csv(patient_range, method_name, prepoc_raw="preproc"):
    """
    Calculate and plot the boxplot of Mean Squared Error (MSE) for a specific method and multiple b-values.

    Parameters:
    - patient_path (str): The name of the patient.
    - method_name (str): Name of the method (e.g., "noddi", "mf", "dti").
    - show (bool): Whether to display the plot using plt.show(). Default is True.
    - save_as_png (bool): Whether to save the plot as a PNG file. Default is False.

    Returns:
    - None
    """
    folder_path = os.path.dirname(os.getcwd())
    writing_path = os.path.join(folder_path, "results/mse", f"mse_values_comparison_{method_name}_by_bval.csv")


    if method_name == "DTI":
        b_values = [1000, 3000]

    elif method_name == "MF":
        b_values = [1000, 3000, 5000]
    else:
        b_values = [1000, 3000, 5000, 10000]



    # Ensure the method list is in the expected order

    with open(writing_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Patient ID'] + b_values +["All shells"])
        
        for patient_id in patient_range:
            patient_path = f"sub-{patient_id}"
            mse_values = []

            if method_name == "MF":
                method_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/"
            else:
                method_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/{method_name}/"



            labels = []

            for bval in b_values:

                if method_name == "DTI":
                    mse, _ = load_nifti(method_path + f"{patient_path}_mse_b{bval}.nii.gz")
                
                elif method_name == "MF":
                    print("MF")
                    mse, _ = load_nifti(method_path+f"{patient_path}_b{bval}_MSE.nii.gz")
                else:
                    mse, _ = load_nifti(method_path + f"{patient_path}_b{bval}_mse.nii.gz")


                brain_mask_path = os.path.join("/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects", patient_path, "masks", f"{patient_path}_wm_mask_AP.nii.gz")
                brain_mask, _ = load_nifti(brain_mask_path)
                brain_mask = np.isin(brain_mask, [1], invert=True)
                
                mse = ma.masked_array(mse, mask=brain_mask).compressed()
                mse = mse[~np.isnan(mse)]

                mse_values.append(mse)

            if method_name == "MF":
                method_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/"
            else:
                method_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/{method_name}/"

            if method_name == "DTI":
                 mse, _ = load_nifti(method_path + f"{patient_path}_mse.nii.gz")
                
            elif method_name == "MF":
                print("MF")
                mse, _ = load_nifti(method_path+f"{patient_path}_MSE.nii.gz")
            else:
                mse, _ = load_nifti(method_path + f"{patient_path}_mse.nii.gz")


            brain_mask_path = os.path.join("/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects", patient_path, "masks", f"{patient_path}_wm_mask_AP.nii.gz")
            brain_mask, _ = load_nifti(brain_mask_path)
            brain_mask = np.isin(brain_mask, [1], invert=True)
                
            mse = ma.masked_array(mse, mask=brain_mask).compressed()
            mse = mse[~np.isnan(mse)]

            mse_values.append(mse)

            nb_combination = len(b_values) +1

            nb_patient = len(mse_values)//nb_combination
            print(nb_patient,len(b_values))
            for i in range(nb_patient):
                nb_element = len(mse_values[i*nb_combination])
                for j in range(nb_element):
                    row = [patient_path]
                    for k in range(nb_combination):
                        row.append(mse_values[i*nb_combination+k][j])
                    writer.writerow(row)

    print(f"MSE values comparison written to {writing_path}")
    return 


def calculate_method_MSE_comparison_by_bvalue_write_csv(patient_range, method_name, prepoc_raw="preproc"):
    """
    Calculates statistical information for Mean Squared Error (MSE) by b-value for a specific MRI method across multiple patients and writes to a CSV file.

    Parameters:
    - patient_range (range): Range of patient IDs to process.
    - method_name (str): Name of the MRI method (e.g., "DTI", "MF", "NODDI").
    - prepoc_raw (str): Specifies the preprocessing stage, default is "preproc".

    Returns:
    - None: Outputs a CSV file with the statistical information.
    """
    folder_path = os.path.dirname(os.getcwd())
    results_path = os.path.join(folder_path, "results/mse")
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    writing_path = os.path.join(results_path, f"mse_stats_comparison_{method_name}_by_bval.csv")

    b_values = {
        "DTI": [1000, 3000],
        "MF": [1000, 3000, 5000],
        "NODDI": [1000, 3000, 5000, 10000]
    }.get(method_name, [1000, 3000, 5000, 10000])

    with open(writing_path, 'w', newline='') as file:
        writer = csv.writer(file)
        headers = ['Patient ID'] + [f'MSE_mean_b{b}' for b in b_values] + [f'MSE_std_b{b}' for b in b_values] + [f'MSE_mean_all'] + [f'MSE_std_all']
        writer.writerow(headers)

        for patient_id in patient_range:
            patient_path = f"sub-{patient_id}"
            stats = [patient_path]
            mse_means = []
            mse_stds = []

            if method_name == "MF":
                method_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/"
            else:
                method_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/{method_name}/"

            for bval in b_values:
                if method_name == "DTI":
                    mse, _ = load_nifti(method_path + f"{patient_path}_mse_b{bval}.nii.gz")
                
                elif method_name == "MF":
                    print("MF")
                    mse, _ = load_nifti(method_path+f"{patient_path}_b{bval}_MSE.nii.gz")
                else:
                    mse, _ = load_nifti(method_path + f"{patient_path}_b{bval}_mse.nii.gz")

                brain_mask_path = os.path.join("/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects", patient_path, "masks", f"{patient_path}_wm_mask_AP.nii.gz")
                brain_mask, _ = load_nifti(brain_mask_path)
                brain_mask = np.isin(brain_mask, [1], invert=True)
                
                filtered_mse = mse[brain_mask]
                mse_means.append(np.nanmean(filtered_mse))
                mse_stds.append(np.nanstd(filtered_mse))
            
            if method_name == "MF":
                method_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/"
            else:
                method_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/{method_name}/"

            if method_name == "DTI":
                 mse, _ = load_nifti(method_path + f"{patient_path}_mse.nii.gz")
                
            elif method_name == "MF":
                print("MF")
                mse, _ = load_nifti(method_path+f"{patient_path}_MSE.nii.gz")
            else:
                mse, _ = load_nifti(method_path + f"{patient_path}_mse.nii.gz")


            brain_mask_path = os.path.join("/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects", patient_path, "masks", f"{patient_path}_wm_mask_AP.nii.gz")
            brain_mask, _ = load_nifti(brain_mask_path)
            brain_mask = np.isin(brain_mask, [1], invert=True)
                
            filtered_mse = mse[brain_mask]
            mse_means.append(np.nanmean(filtered_mse))
            mse_stds.append(np.nanstd(filtered_mse))


            # Append mean and std for each b-value
            stats += mse_means + mse_stds
            writer.writerow(stats)

    print(f"MSE values comparison written to {writing_path}")


################################################################### BVALUE #############################################################

def compare_MRI_bvalues_to_csv(patient_path, method_name, data_type, preprocessing_type, save_csv = True):
    #brain_mask_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/masks/{patient_path}_brain_mask.nii.gz"
    brain_mask_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/masks/{patient_path}_wm_mask_AP.nii.gz"

    brain_mask, _ = load_nifti(brain_mask_path)

    brain_mask_mask = brain_mask == 1

    if method_name == "DTI":
        b_values = [1000, 3000]

    elif method_name == "MF":
        b_values = [1000, 3000, 5000]
    else:
        b_values = [1000, 3000, 5000, 10000]

    data_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/{method_name}/"
    
    filtered_data = {}

    for b_value in b_values:
        if method_name == "noddi":
            data, _ = load_nifti(data_path + f"{patient_path}_b{b_value}_{data_type}.nii.gz")
        elif method_name == "MF":
            data, _ = load_nifti(f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/{patient_path}_b{b_value}_{data_type}.nii.gz")
        else:
            data, _ = load_nifti(data_path + f"{patient_path}_{data_type}_b{b_value}.nii.gz")
        
        print(f"Loading data from: {data_path}")

        filtered_data[b_value] = data[brain_mask_mask]

    if method_name =="MF":
        data, _ = load_nifti(f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/{patient_path}_{data_type}.nii.gz")
    else:   
        data, _ = load_nifti(data_path + f"{patient_path}_{data_type}.nii.gz")
    filtered_data["all bval"] = data[brain_mask_mask]
    stats = []

    for b_value, data in filtered_data.items():

        stats.append({
            "b_value": b_value,
            "mean": np.mean(data),
            "median": np.median(data),
            "std_dev": np.std(data),
            "min": np.min(data),
            "max": np.max(data)
        })
    df = pd.DataFrame(stats)

    if save_csv:
        folder_path = os.path.abspath(os.path.dirname(os.getcwd()))
        save_path = folder_path + f"/results/{method}/stats_comparison_{patient_path}_{method_name}_{data_type}.csv"
        df.to_csv(save_path, index=False)

    return  


def compare_MRI_bvalues_to_csv_ROI(patient_path, method_name, data_type, atlas_name,roi_name,atlas_values, save_csv = True):
    #brain_mask_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/masks/{patient_path}_brain_mask.nii.gz"
    subject_folder = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}"

    brain_mask_path = f"{subject_folder}/reg/{patient_path}_Atlas_{atlas_name}_InSubjectDWISpaceFrom_AP.nii.gz"
    brain_mask, _ = load_nifti(brain_mask_path)
    brain_mask_mask = np.where(np.isin(brain_mask, atlas_values), 0, 1)

    if method_name == "DTI":
        b_values = [1000, 3000]

    elif method_name == "MF":
        b_values = [1000, 3000, 5000]
    else:
        b_values = [1000, 3000, 5000, 10000]

    data_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/{method_name}/"
    
    filtered_data = {}

    for b_value in b_values:
        if method_name == "noddi":
            data, _ = load_nifti(data_path + f"{patient_path}_b{b_value}_{data_type}.nii.gz")
        elif method_name == "MF":
            data, _ = load_nifti(f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/{patient_path}_b{b_value}_{data_type}.nii.gz")
        else:
            data, _ = load_nifti(data_path + f"{patient_path}_{data_type}_b{b_value}.nii.gz")
        
        print(f"Loading data from: {data_path}")

        filtered_data[b_value] = ma.masked_array(data, mask=brain_mask_mask).compressed()

    if method_name =="MF":
        data, _ = load_nifti(f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/{patient_path}_{data_type}.nii.gz")
    else:   
        data, _ = load_nifti(data_path + f"{patient_path}_{data_type}.nii.gz")
    filtered_data["all bval"] = ma.masked_array(data, mask=brain_mask_mask).compressed()
    stats = []

    for b_value, data in filtered_data.items():

        stats.append({
            "b_value": b_value,
            "mean": np.mean(data),
            "median": np.median(data),
            "std_dev": np.std(data),
            "min": np.min(data),
            "max": np.max(data)
        })
    df = pd.DataFrame(stats)

    if save_csv:
        folder_path = os.path.abspath(os.path.dirname(os.getcwd()))
        save_path = folder_path + f"/results/{method_name}/{roi_name}/stats_comparison_{roi_name}_{patient_path}_{method_name}_{data_type}.csv"
        print(save_path)
        df.to_csv(save_path, index=False)

    return  



def calculate_bval_write_csv_ROI(patient_range, method_name, data_type, atlas_name, atlas_values, roi_name):
    """
    Writes data values for a specified MRI method and data type across a range of patients into a CSV file,
    with columns for patient ID followed by data values for each specified b-value.

    Args:
    - patient_range (range): Range of patient IDs to process.
    - method_name (str): MRI method name (e.g., "DTI", "noddi", "MF").
    - data_type (str): Type of data being written (e.g., "FA", "MD").

    Returns:
    - None: Outputs a CSV file with data values.
    """
    folder_path = os.path.dirname(os.getcwd())
    writing_path = os.path.join(folder_path, "results", method_name,roi_name, f"{method_name}_{data_type}_values_comparison_{roi_name}.csv")

    os.makedirs(os.path.dirname(writing_path), exist_ok=True)

    if method_name == "DTI":
        b_values = [1000, 3000]

    elif method_name == "MF":
        b_values = [1000, 3000, 5000]
    else:
        b_values = [1000, 3000, 5000, 10000]

    with open(writing_path, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the header with b-values
        writer.writerow(['Patient ID'] + [f"b{b_value}" for b_value in b_values]+["all_bvals"])
        
        for patient_id in patient_range:
            patient_path = f"sub-{patient_id}"
            subject_folder = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}"
            data_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/{method_name}/"
            print(f"Processing patient {patient_path} for {method_name} using {data_type}")
            data_values = []

            brain_mask_path = f"{subject_folder}/reg/{patient_path}_Atlas_{atlas_name}_InSubjectDWISpaceFrom_AP.nii.gz"
            brain_mask, _ = load_nifti(brain_mask_path)
            brain_mask_mask = np.where(np.isin(brain_mask, atlas_values), 0, 1)

            
            for b_value in b_values:
                if method_name == "noddi":
                    data, _ = load_nifti(data_path + f"{patient_path}_b{b_value}_{data_type}.nii.gz")
                elif method_name == "MF":
                    data, _ = load_nifti(f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/{patient_path}_b{b_value}_{data_type}.nii.gz")
                else:
                    data, _ = load_nifti(data_path + f"{patient_path}_{data_type}_b{b_value}.nii.gz")

                 
                data = ma.masked_array(data, mask=brain_mask_mask).compressed()
                data = data[~np.isnan(data)]

                data_values.append(data)
                print(f"Loading data for b-value {b_value}")

            
            if method_name == "noddi":
                data, _ = load_nifti(data_path + f"{patient_path}_{data_type}.nii.gz")
            elif method_name == "MF":
                data, _ = load_nifti(f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/{patient_path}_{data_type}.nii.gz")
            else:
                data, _ = load_nifti(data_path + f"{patient_path}_{data_type}.nii.gz")

            data = ma.masked_array(data, mask=brain_mask_mask).compressed()
            data = data[~np.isnan(data)]

            data_values.append(data)
                
            len_reduced = len(b_values) +1

            row = [patient_path]
            for z in range(len_reduced):
                row.append(np.nanmean(data_values[z]))    
            writer.writerow(row)
        print(f"Data values comparison written to {writing_path}")
                
    print(f"Data values comparison for {method_name} written to {writing_path}")


def calculate_bval_write_csv(patient_range, method_name, data_type):
    """
    Writes data values for a specified MRI method and data type across a range of patients into a CSV file,
    with columns for patient ID followed by data values for each specified b-value.

    Args:
    - patient_range (range): Range of patient IDs to process.
    - method_name (str): MRI method name (e.g., "DTI", "noddi", "MF").
    - data_type (str): Type of data being written (e.g., "FA", "MD").

    Returns:
    - None: Outputs a CSV file with data values.
    """
    folder_path = os.path.dirname(os.getcwd())
    writing_path = os.path.join(folder_path, "results", method_name, f"{method_name}_{data_type}_values_comparison.csv")

    # Ensure the results directory exists
    os.makedirs(os.path.dirname(writing_path), exist_ok=True)

    # Define b-values based on the MRI method
    if method_name == "DTI":
        b_values = [1000, 3000]
    elif method_name == "MF":
        b_values = [1000, 3000, 5000]
    else:
        b_values = [1000, 3000, 5000, 10000]

    with open(writing_path, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the header with b-values
        writer.writerow(['Patient ID'] + [f"b{b_value}" for b_value in b_values]+["all_bvals"])
        
        for patient_id in patient_range:
            patient_path = f"sub-{patient_id}"
            data_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/{method_name}/"
            print(f"Processing patient {patient_path} for {method_name} using {data_type}")
            data_values = []

            brain_mask_path = os.path.join("/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects", patient_path, "masks", f"{patient_path}_wm_mask_AP.nii.gz")
            brain_mask, _ = load_nifti(brain_mask_path)
            brain_mask = np.isin(brain_mask, [1], invert=True)
            
            for b_value in b_values:
                if method_name == "noddi":
                    data, _ = load_nifti(data_path + f"{patient_path}_b{b_value}_{data_type}.nii.gz")
                elif method_name == "MF":
                    data, _ = load_nifti(f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/{patient_path}_b{b_value}_{data_type}.nii.gz")
                else:
                    data, _ = load_nifti(data_path + f"{patient_path}_{data_type}_b{b_value}.nii.gz")

                 
                data = ma.masked_array(data, mask=brain_mask).compressed()
                data = data[~np.isnan(data)]

                data_values.append(data)
                print(f"Loading data for b-value {b_value}")

            
            if method_name == "noddi":
                data, _ = load_nifti(data_path + f"{patient_path}_{data_type}.nii.gz")
            elif method_name == "MF":
                data, _ = load_nifti(f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/{patient_path}_{data_type}.nii.gz")
            else:
                data, _ = load_nifti(data_path + f"{patient_path}_{data_type}.nii.gz")

            data = ma.masked_array(data, mask=brain_mask).compressed()
            data = data[~np.isnan(data)]

            data_values.append(data)
                
            len_reduced = len(b_values) +1

            row = [patient_path]
            for z in range(len_reduced):
                    row.append(np.nanmean(data_values[z]))
                        
            writer.writerow(row)
        print(f"Data values comparison written to {writing_path}")
                
    print(f"Data values comparison for {method_name} written to {writing_path}")



################################################################### REDUCED #############################################################
def comparison_reduced_write_csv(patient_range, method_name, data_type, reduced_bval, reduced_nb):
    """
    Writes statistical data values for a specified MRI method and data type across a range of patients into a CSV file,
    with columns for patient ID and values for each specified reduction in b-values.

    Args:
    - patient_range (range): Range of patient IDs to process.
    - method_name (str): MRI method name (e.g., "dti", "noddi", "mf").
    - data_type (str): Type of data being written (e.g., "FA", "MD").
    - reduced_bval (int): Specific b-value that has been reduced.
    - reduced_nb (list): List of numbers specifying the reduction levels.

    Returns:
    - None: Outputs a CSV file with data values.
    """
    folder_path = os.path.dirname(os.getcwd())
    results_path = os.path.join(folder_path, "results", method_name, str(reduced_bval))
    os.makedirs(results_path, exist_ok=True)
    writing_path = os.path.join(results_path, f"{method_name}_{data_type}_stats_comparison_b{reduced_bval}.csv")

    with open(writing_path, 'w', newline='') as file:
        writer = csv.writer(file)
        header = ['Patient ID']
        for nb in reduced_nb:
            header += [f"{nb}_mean", f"{nb}_median", f"{nb}_std", f"{nb}_min", f"{nb}_max"]
        header += ["Original_mean", "Original_median", "Original_std", "Original_min", "Original_max"]
        writer.writerow(header)
        
        for patient_id in patient_range:
            patient_path = f"sub-{patient_id}"
            print(f"Processing patient: {patient_path}")
            data_values = []

            brain_mask_path = os.path.join("/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects", patient_path, "masks", f"{patient_path}_wm_mask_AP.nii.gz")
            brain_mask, _ = load_nifti(brain_mask_path)
            brain_mask = np.isin(brain_mask, [1], invert=True)
            
            stats = [patient_path]
            
            for k in reduced_nb:
                data_path = f"{patient_path}_reduced_b{reduced_bval}_{k}_{data_type}.nii.gz"
                if method_name == "MF":
                    full_path = os.path.join("/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects", patient_path, "dMRI/microstructure/mf", data_path)
                else:
                    full_path = os.path.join("/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects", patient_path, "dMRI", method_name, data_path)
                
                data, _ = load_nifti(full_path)
                data = ma.masked_array(data, mask=brain_mask).compressed()
                data = data[~np.isnan(data)]
                data_values.append(data)

                # Calculate stats
                stats += [np.mean(data), np.median(data), np.std(data), np.min(data), np.max(data)]

            # Calculate stats for original data without reduction
            data_path = f"{patient_path}_{data_type}.nii.gz"
            if method_name == "MF":
                full_path = os.path.join("/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects", patient_path, "dMRI/microstructure/mf", data_path)
            else:
                full_path = os.path.join("/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects", patient_path, "dMRI", method_name, data_path)

            data, _ = load_nifti(full_path)
            data = ma.masked_array(data, mask=brain_mask).compressed()
            data = data[~np.isnan(data)]

            stats += [np.mean(data), np.median(data), np.std(data), np.min(data), np.max(data)]
            writer.writerow(stats)

    print(f"Data values comparison written to {writing_path}")




def calculate_reduced_write_csv(patient_range, method_name, data_type, reduced_bval, reduced_nb):
    """
    Writes data values for a specified MRI method and data type across a range of patients into a CSV file,
    with columns for patient ID and values for each specified reduction in b-values.

    Args:
    - patient_range (range): Range of patient IDs to process.
    - method_name (str): MRI method name (e.g., "dti", "noddi", "mf").
    - data_type (str): Type of data being written (e.g., "FA", "MD").
    - reduced_bval (int): Specific b-value that has been reduced.
    - reduced_nb (list): List of numbers specifying the reduction levels.

    Returns:
    - None: Outputs a CSV file with data values.
    """
    folder_path = os.path.dirname(os.getcwd())
    writing_path = os.path.join(folder_path, "results", method_name,str(reduced_bval), f"{method_name}_{data_type}_values_comparison_b{reduced_bval}.csv")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(writing_path), exist_ok=True)

    if reduced_bval < 5000:
        nb_direction_original = 64

    elif reduced_bval == 5000:
        nb_direction_original = 128
    
    else:
        nb_direction_original = 256
    

    with open(writing_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Patient ID'] + [f"{nb}" for nb in reduced_nb] + [str(nb_direction_original)])
        
        for patient_id in patient_range:
            patient_path = f"sub-{patient_id}"
            print(f"Processing patient: {patient_path}")
            data_values = []

            brain_mask_path = os.path.join("/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects", patient_path, "masks", f"{patient_path}_wm_mask_AP.nii.gz")
            brain_mask, _ = load_nifti(brain_mask_path)
            brain_mask = np.isin(brain_mask, [1], invert=True)
            
            for k in reduced_nb:
                print(f"Loading data for {method_name}, b-value: {reduced_bval}, reduction: {k}")
                data_path = f"{patient_path}_reduced_b{reduced_bval}_{k}_{data_type}.nii.gz"
                if method_name == "MF":
                    method_path = os.path.join("/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects", patient_path, "dMRI/microstructure/mf", data_path)
                else:
                    method_path = os.path.join("/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects", patient_path, "dMRI", method_name, data_path)
                
                data, _ = load_nifti(method_path)
            
                data = ma.masked_array(data, mask=brain_mask).compressed()
                data = data[~np.isnan(data)]

                data_values.append(data)

            if method_name == "noddi":
                data, _ = load_nifti(os.path.join("/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects", patient_path, "dMRI", method_name, f"{patient_path}_{data_type}.nii.gz"))
            elif method_name == "MF":
                data, _ = load_nifti(f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/{patient_path}_{data_type}.nii.gz")
            else:
                data, _ = load_nifti(os.path.join("/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects", patient_path, "dMRI", method_name, f"{patient_path}_{data_type}.nii.gz"))

            data = ma.masked_array(data, mask=brain_mask).compressed()
            data = data[~np.isnan(data)]

            data_values.append(data)

            len_reduced = len(reduced_nb) +1

            row = [patient_path]
            for z in range(len_reduced):
                row.append(np.nanmean(data_values[z]))
                        
            writer.writerow(row)
        print(f"Data values comparison written to {writing_path}")

    return 


def calculate_reduced_write_csv_ROI(patient_range, method_name, data_type, reduced_bval, reduced_nb,atlas_name, atlas_values, roi_name):
    """
    Writes data values for a specified MRI method and data type across a range of patients into a CSV file,
    with columns for patient ID and values for each specified reduction in b-values.

    Args:
    - patient_range (range): Range of patient IDs to process.
    - method_name (str): MRI method name (e.g., "dti", "noddi", "mf").
    - data_type (str): Type of data being written (e.g., "FA", "MD").
    - reduced_bval (int): Specific b-value that has been reduced.
    - reduced_nb (list): List of numbers specifying the reduction levels.

    Returns:
    - None: Outputs a CSV file with data values.
    """
    folder_path = os.path.dirname(os.getcwd())
    writing_path = os.path.join(folder_path, "results", method_name,roi_name,str(reduced_bval), f"{method_name}_{data_type}_values_comparison_b{reduced_bval}.csv")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(writing_path), exist_ok=True)

    if reduced_bval < 5000:
        nb_direction_original = 64

    elif reduced_bval == 5000:
        nb_direction_original = 128
    else:
        nb_direction_original = 256

    with open(writing_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Patient ID'] + [f"{nb}" for nb in reduced_nb]+ [str(nb_direction_original)])
        
        for patient_id in patient_range:
            patient_path = f"sub-{patient_id}"
            subject_folder = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}"
            print(f"Processing patient: {patient_path}")
            data_values = []

            brain_mask_path = f"{subject_folder}/reg/{patient_path}_Atlas_{atlas_name}_InSubjectDWISpaceFrom_AP.nii.gz"
            brain_mask, _ = load_nifti(brain_mask_path)
            brain_mask_mask = np.where(np.isin(brain_mask, atlas_values), 0, 1)
            
            for k in reduced_nb:
                print(f"Loading data for {method_name}, b-value: {reduced_bval}, reduction: {k}")
                data_path = f"{patient_path}_reduced_b{reduced_bval}_{k}_{data_type}.nii.gz"
                if method_name == "MF":
                    method_path = os.path.join("/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects", patient_path, "dMRI/microstructure/mf", data_path)
                else:
                    method_path = os.path.join("/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects", patient_path, "dMRI", method_name, data_path)
                
                data, _ = load_nifti(method_path)
                
                data = ma.masked_array(data, mask=brain_mask_mask).compressed()
                data = data[~np.isnan(data)]

                data_values.append(data)
            
            if method_name == "noddi":
                data, _ = load_nifti(os.path.join("/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects", patient_path, "dMRI", method_name, f"{patient_path}_{data_type}.nii.gz"))
            elif method_name == "MF":
                data, _ = load_nifti(f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/{patient_path}_{data_type}.nii.gz")
            else:
                data, _ = load_nifti(os.path.join("/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects", patient_path, "dMRI", method_name, f"{patient_path}_{data_type}.nii.gz"))

            data = ma.masked_array(data, mask=brain_mask_mask).compressed()
            data = data[~np.isnan(data)]

            data_values.append(data)

            len_reduced = len(reduced_nb) +1

            row = [patient_path]
            for z in range(len_reduced):
                row.append(np.nanmean(data_values[z]))
                        
            writer.writerow(row)
        print(f"Data values comparison written to {writing_path}")

    return 
################################################################### PAIR #############################################################

def compare_MRI_pair_to_csv(patient_path, method_name, data_type, save_csv = True):
    brain_mask_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/masks/{patient_path}_brain_mask.nii.gz"
    #brain_mask_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/masks/{patient_path}_wm_mask_AP.nii.gz"

    brain_mask, _ = load_nifti(brain_mask_path)

    brain_mask_mask = brain_mask == 1

    if method_name == "DTI":
        b_values = [1000, 3000]

    else:
        b_values = [1000, 3000, 5000, 10000]

    data_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/{method}/"
    
    filtered_data = {}

    bval_combinations = [(b1, b2) for i, b1 in enumerate(b_values) for b2 in b_values[i+1:] if b1 != b2]

    labels = []
    for bval_pair in bval_combinations:

        bval_str = f"b{bval_pair[0]}b{bval_pair[1]}"
        labels.append(bval_str)
        if method_name == "noddi":
            data, _ = load_nifti(data_path + f"{patient_path}_{bval_str}_{data_type}.nii.gz")
        elif method_name == "MF":
            data, _ = load_nifti(f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/{patient_path}_{bval_str}_{data_type}.nii.gz")
        else:
            data, _ = load_nifti(data_path + f"{patient_path}_{data_type}_{bval_str}.nii.gz")
        
        print(f"Loading data from: {data_path}")

        filtered_data[bval_str] = data[brain_mask_mask]

    if method_name =="MF":
        data, _ = load_nifti(f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/{patient_path}_{data_type}.nii.gz")
    else:   
        data, _ = load_nifti(data_path + f"{patient_path}_{data_type}.nii.gz")
    filtered_data["all bval"] = data[brain_mask_mask]
    stats = []

    for b_value, data in filtered_data.items():

        stats.append({
            "b_value": b_value,
            "mean": np.mean(data),
            "median": np.median(data),
            "std_dev": np.std(data),
            "min": np.min(data),
            "max": np.max(data)
        })
    df = pd.DataFrame(stats)

    if save_csv:
        folder_path = os.path.abspath(os.path.dirname(os.getcwd()))
        save_path = folder_path + f"/results/{method_name}/pairs/stats_comparison_pair_{patient_path}_{method_name}_{data_type}.csv"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)

    return  



def calculate_bval_pair_write_csv_ROI(patient_range, method_name, data_type,atlas_name, atlas_values, roi_name):
    """
    Writes data values for a specified MRI method and data type across a range of patients into a CSV file,
    with columns for patient ID followed by data values for each specified b-value.

    Args:
    - patient_range (range): Range of patient IDs to process.
    - method_name (str): MRI method name (e.g., "DTI", "noddi", "MF").
    - data_type (str): Type of data being written (e.g., "FA", "MD").

    Returns:
    - None: Outputs a CSV file with data values.
    """
    folder_path = os.path.dirname(os.getcwd())
    writing_path = os.path.join(folder_path, "results", method_name,roi_name, "pair", f"{method_name}_{data_type}_values_comparison_pair_{roi_name}.csv")

    # Ensure the results directory exists
    os.makedirs(os.path.dirname(writing_path), exist_ok=True)

    # Define b-values based on the MRI method
    if method_name == "DTI":
        b_values = [1000, 3000]
    else:
        b_values = [1000, 3000, 5000, 10000]

    bval_combinations = [(b1, b2) for i, b1 in enumerate(b_values) for b2 in b_values[i+1:] if b1 != b2]

    with open(writing_path, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the header with b-values
        writer.writerow(['Patient ID'] + [f"{b_value}" for b_value in bval_combinations]+["all_bvals"])
        
        for patient_id in patient_range:
            patient_path = f"sub-{patient_id}"
            subject_folder = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}"
            data_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/{method_name}/"
            print(f"Processing patient {patient_path} for {method_name} using {data_type}")
            data_values = []

            brain_mask_path = f"{subject_folder}/reg/{patient_path}_Atlas_{atlas_name}_InSubjectDWISpaceFrom_AP.nii.gz"
            brain_mask, _ = load_nifti(brain_mask_path)
            brain_mask_mask = np.where(np.isin(brain_mask, atlas_values), 0, 1)
            
            for bval_pair in bval_combinations:
                bval_str = f"b{bval_pair[0]}b{bval_pair[1]}"
                if method_name == "noddi":
                    data, _ = load_nifti(data_path + f"{patient_path}_{bval_str}_{data_type}.nii.gz")
                elif method_name == "MF":
                    data, _ = load_nifti(f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/{patient_path}_{bval_str}_{data_type}.nii.gz")
                else:
                    data, _ = load_nifti(data_path + f"{patient_path}_{data_type}_{bval_str}.nii.gz")

                 
                data = ma.masked_array(data, mask=brain_mask_mask).compressed()
                data = data[~np.isnan(data)]

                data_values.append(np.mean(data))
                print(f"Loading data for b-value {bval_str}")

            
            if method_name == "noddi":
                data, _ = load_nifti(data_path + f"{patient_path}_{data_type}.nii.gz")
            elif method_name == "MF":
                data, _ = load_nifti(f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/{patient_path}_{data_type}.nii.gz")
            else:
                data, _ = load_nifti(data_path + f"{patient_path}_{data_type}.nii.gz")

            data = ma.masked_array(data, mask=brain_mask_mask).compressed()
            data = data[~np.isnan(data)]

            data_values.append(np.mean(data))

            print(data_values)
                
            len_reduced = len(bval_combinations) +1

            row = [patient_path]
            for z in range(len_reduced):
                row.append(data_values[z])
                print(row)       
            writer.writerow(row)

        print(f"Data values comparison written to {writing_path}")
                
    print(f"Data values comparison for {method_name} written to {writing_path}")
    
def calculate_bval_pair_write_csv(patient_range, method_name, data_type):
    """
    Writes data values for a specified MRI method and data type across a range of patients into a CSV file,
    with columns for patient ID followed by data values for each specified b-value.

    Args:
    - patient_range (range): Range of patient IDs to process.
    - method_name (str): MRI method name (e.g., "DTI", "noddi", "MF").
    - data_type (str): Type of data being written (e.g., "FA", "MD").

    Returns:
    - None: Outputs a CSV file with data values.
    """
    folder_path = os.path.dirname(os.getcwd())
    writing_path = os.path.join(folder_path, "results", method_name,"pair", f"{method_name}_{data_type}_values_comparison_pair.csv")

    # Ensure the results directory exists
    os.makedirs(os.path.dirname(writing_path), exist_ok=True)

    # Define b-values based on the MRI method
    if method_name == "DTI":
        b_values = [1000, 3000]
    else:
        b_values = [1000, 3000, 5000, 10000]

    bval_combinations = [(b1, b2) for i, b1 in enumerate(b_values) for b2 in b_values[i+1:] if b1 != b2]

    with open(writing_path, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the header with b-values
        writer.writerow(['Patient ID'] + [f"{b_value}" for b_value in bval_combinations]+["all_bvals"])
        
        for patient_id in patient_range:
            patient_path = f"sub-{patient_id}"
            data_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/{method_name}/"
            print(f"Processing patient {patient_path} for {method_name} using {data_type}")
            data_values = []

            brain_mask_path = os.path.join("/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects", patient_path, "masks", f"{patient_path}_wm_mask_AP.nii.gz")
            brain_mask, _ = load_nifti(brain_mask_path)
            brain_mask = np.isin(brain_mask, [1], invert=True)
            
            for bval_pair in bval_combinations:
                bval_str = f"b{bval_pair[0]}b{bval_pair[1]}"
                if method_name == "noddi":
                    data, _ = load_nifti(data_path + f"{patient_path}_{bval_str}_{data_type}.nii.gz")
                elif method_name == "MF":
                    data, _ = load_nifti(f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/{patient_path}_{bval_str}_{data_type}.nii.gz")
                else:
                    data, _ = load_nifti(data_path + f"{patient_path}_{data_type}_{bval_str}.nii.gz")

                 
                data = ma.masked_array(data, mask=brain_mask).compressed()
                data = data[~np.isnan(data)]

                data_values.append(data)
                print(f"Loading data for b-value {bval_str}")

            
            if method_name == "noddi":
                data, _ = load_nifti(data_path + f"{patient_path}_{data_type}.nii.gz")
            elif method_name == "MF":
                data, _ = load_nifti(f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/{patient_path}_{data_type}.nii.gz")
            else:
                data, _ = load_nifti(data_path + f"{patient_path}_{data_type}.nii.gz")

            data = ma.masked_array(data, mask=brain_mask).compressed()
            data = data[~np.isnan(data)]

            data_values.append(data)
                
            len_reduced = len(bval_combinations) +1


            row = [patient_path]
            for z in range(len_reduced):
                    row.append(np.nanmean(data_values[z]))
                        
            writer.writerow(row)
        print(f"Data values comparison written to {writing_path}")
                
    print(f"Data values comparison for {method_name} written to {writing_path}")

#########################################################################################################################################################################################################

patient_range = range(1001, 1011)


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

#method ="DTI"
#method ="noddi"
method ="MF"


method_list = ["DTI","noddi","MF"]

parameters_dti = ["FA","MD","RD","AD"]
parameters_noddi = ["fbundle","fextra", "fintra","fiso", "icvf", "odi"]#mu
parameters_mf = ["fvf_f0","fvf_f1","fvf_tot", "frac_csf","frac_f0","frac_f1","DIFF_ex_f0","DIFF_ex_f1","DIFF_ex_tot"]


atlas_name = "JHU-ICBM-labels-1mm"

atlas_values = [5]
roi_name ="body_of_corpus_callosum"



for i in range(1001,1007):
    patient_path = f"sub-{i}"

    for parameter in parameters_mf:

        #compare_MRI_bvalues_to_csv(patient_path, method, parameter, preprocessing_type='prepoc', save_csv = True)
        #compare_MRI_pair_to_csv(patient_path, method, parameter, save_csv = True)
        #comparison_reduced_write_csv(patient_range, method, parameter, 1000, [16,32,40,48])
        pass
    



reduced_bval = 5000
reduced_nb=[32, 64, 100]


for parameter in parameters_dti:
    pass

    #calculate_bval_write_csv(patient_range, "DTI", parameter)


    #calculate_bval_pair_write_csv_ROI(patient_range, "DTI", parameter,atlas_name, atlas_values, roi_name)
    
    #calculate_bval_pair_write_csv(patient_range, "DTI", parameter)

    #calculate_reduced_write_csv(patient_range, "DTI", parameter, reduced_bval, reduced_nb)
    #calculate_reduced_write_csv_ROI(patient_range, "DTI", parameter, reduced_bval, reduced_nb,atlas_name, atlas_values, roi_name)

    #calculate_bval_write_csv_ROI(patient_range, "DTI", parameter, atlas_name, atlas_values, roi_name)

    #comparison_reduced_write_csv(patient_range, 'DTI', parameter, reduced_bval, reduced_nb)



for parameter in parameters_noddi:
    pass

    #calculate_bval_write_csv(patient_range, "noddi", parameter)
    
    #calculate_bval_pair_write_csv_ROI(patient_range, "noddi", parameter,atlas_name, atlas_values, roi_name)

    #calculate_bval_pair_write_csv(patient_range, "noddi", parameter)
    #calculate_reduced_write_csv(patient_range, "noddi", parameter, reduced_bval, reduced_nb)

    #calculate_bval_write_csv_ROI(patient_range, "noddi", parameter, atlas_name, atlas_values, roi_name)

    #comparison_reduced_write_csv(patient_range, 'noddi', parameter, reduced_bval, reduced_nb)





for parameter in parameters_mf:
    pass
    #calculate_bval_write_csv(patient_range, "MF", parameter)

    #calculate_bval_pair_write_csv_ROI(patient_range, "MF", parameter,atlas_name, atlas_values, roi_name)
    #calculate_reduced_write_csv(patient_range, "MF", parameter, reduced_bval, reduced_nb)
    #comparison_reduced_write_csv(patient_range, 'MF', parameter, reduced_bval, reduced_nb)
    #calculate_bval_pair_write_csv(patient_range, "MF", parameter)

    #calculate_bval_write_csv_ROI(patient_range, "MF", parameter, atlas_name, atlas_values, roi_name)





reduced_bval = 5000
reduced_nb=[32, 64, 100]


for key, value in tract_dictionary.items():
    pass
    #calculate_mse_multi_methods_write_csv_ROI(range(1001, 1011), method_list,atlas_name, [key], value)
    for parameter in parameters_mf:
        
        #calculate_bval_write_csv_ROI(patient_range, "MF", parameter, atlas_name, [key], value)
        #calculate_bval_pair_write_csv_ROI(patient_range, "MF", parameter,atlas_name,[key], value)
        calculate_reduced_write_csv_ROI(patient_range, "MF", parameter, reduced_bval, reduced_nb,atlas_name, [key], value)

        #compare_MRI_bvalues_to_csv_ROI("sub-1001", "MF", parameter, atlas_name,value,[key], save_csv = True)

    for parameter in parameters_noddi:
        pass
        #calculate_bval_write_csv_ROI(patient_range, "noddi", parameter, atlas_name, [key], value)
        #calculate_bval_pair_write_csv_ROI(patient_range, "noddi", parameter,atlas_name,[key], value)
        #calculate_reduced_write_csv_ROI(patient_range, "noddi", parameter, reduced_bval, reduced_nb,atlas_name, [key], value)
        #compare_MRI_bvalues_to_csv_ROI("sub-1001", "noddi", parameter, atlas_name,value,[key], save_csv = True)

    for parameter in parameters_dti:
        pass
        #calculate_bval_write_csv_ROI(patient_range, "DTI", parameter, atlas_name, [key], value)
        #calculate_bval_pair_write_csv_ROI(patient_range, "DTI", parameter,atlas_name,[key], value)
        #calculate_reduced_write_csv_ROI(patient_range, "DTI", parameter, reduced_bval, reduced_nb,atlas_name, [key], value)

        #compare_MRI_bvalues_to_csv_ROI("sub-1001", "DTI", parameter, atlas_name,value,[key], save_csv = True)




#calculate_method_MSE_reduced_write_csv(patient_range, "DTI",1000, prepoc_raw="preproc")

#calculate_method_MSE_reduced_write_csv(patient_range, "DTI",3000, prepoc_raw="preproc")



#calculate_method_MSE_reduced_write_csv(patient_range, "noddi",1000, prepoc_raw="preproc")

#calculate_method_MSE_reduced_write_csv(patient_range, "noddi",3000, prepoc_raw="preproc")

#calculate_method_MSE_reduced_write_csv(patient_range, "noddi",5000, prepoc_raw="preproc")

#calculate_method_MSE_reduced_write_csv(patient_range, "noddi",10000, prepoc_raw="preproc")


#calculate_method_MSE_reduced_write_csv(patient_range, "MF",1000, prepoc_raw="preproc")

#calculate_method_MSE_reduced_write_csv(patient_range, "MF",3000, prepoc_raw="preproc")

#calculate_method_MSE_reduced_write_csv(patient_range, "MF",5000, prepoc_raw="preproc")

#calculate_method_MSE_reduced_write_csv(patient_range, "MF",10000, prepoc_raw="preproc")



