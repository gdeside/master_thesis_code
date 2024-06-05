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


def comparison_scatter_hist(patient_path,atlas_name, atlas_values, roi_name, method_name_x,method_name_y, data_type_x,data_type_y, show=False, save_as_png=False):
    
    subject_folder = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}"

    brain_mask_path = f"{subject_folder}/reg/{patient_path}_Atlas_{atlas_name}_InSubjectDWISpaceFrom_AP.nii.gz"

    brain_mask, _ = load_nifti(brain_mask_path)

    brain_mask_mask = np.where(np.isin(brain_mask, atlas_values), 0, 1)

    b_values_x = [1000, 3000] if method_name_x == "DTI" else [1000, 3000, 5000, 10000]
    b_values_y = [1000, 3000] if method_name_y == "DTI" else [1000, 3000, 5000, 10000]
    if method_name_x == "MF" or method_name_y == "MF":
        b_values_x = b_values_y = [1000, 3000, 5000]  # Assuming 'MF' method uses these b-values

    # Determine the common b-values to process
    b_values = list(set(b_values_x).intersection(b_values_y))

    data_path_x = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/{method_name_x}/"

    data_path_y = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/{method_name_y}/"

    print(b_values)
    for bval in b_values:
        print(bval)
        if method_name_x == "noddi":
            data_x, _ = load_nifti(data_path_x + f"{patient_path}_b{bval}_{data_type_x}.nii.gz")
        elif method_name_x == "MF":
            data_x, _ = load_nifti(f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/{patient_path}_b{bval}_{data_type_x}.nii.gz")
        else:
            data_x, _ = load_nifti(data_path_x + f"{patient_path}_{data_type_x}_b{bval}.nii.gz")

        data_x = ma.masked_array(data_x, mask=brain_mask_mask)

        x = data_x.compressed()


        if method_name_y == "noddi":
            data_y, _ = load_nifti(data_path_y + f"{patient_path}_b{bval}_{data_type_y}.nii.gz")
        elif method_name_y == "MF":
            data_y, _ = load_nifti(f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/{patient_path}_b{bval}_{data_type_y}.nii.gz")
        else:
            data_y, _ = load_nifti(data_path_y + f"{patient_path}_{data_type_y}_b{bval}.nii.gz")

        data_y = ma.masked_array(data_y, mask=brain_mask_mask)

        y = data_y.compressed()
        
   
        gs = gridspec.GridSpec(3, 3)
        ax_main = plt.subplot(gs[1:3, :2])
        ax_xDist = plt.subplot(gs[0, :2],sharex=ax_main)
        ax_yDist = plt.subplot(gs[1:3, 2],sharey=ax_main)
            
        ax_main.scatter(x,y,marker='.')
        ax_main.set(xlabel=data_type_x, ylabel=data_type_y)

        ax_xDist.hist(x,bins=100,align='mid')
        ax_xDist.set(ylabel='count')
        
        ax_yDist.hist(y,bins=100,orientation='horizontal',align='mid')
        ax_yDist.set(xlabel='count')

        folder_path = os.path.abspath(os.path.dirname(os.getcwd()))

        save_path = folder_path + f"/images/{method_name_x}/scatter_comparison_{roi_name}_{patient_path}_{method_name_x}_{data_type_x}_{method_name_y}_{data_type_y}_b{bval}.png"
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        if save_as_png:
            plt.savefig(save_path)
            print(f"Plot saved as PNG: {save_path}")
        if show:
            plt.show()
        plt.close()
    return



def comparison_with_allshell_scatter_hist(patient_path,atlas_name, atlas_values, roi_name, method_name_x,method_name_y, data_type_x,data_type_y, show=False, save_as_png=False):
    
    subject_folder = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}"

    brain_mask_path = f"{subject_folder}/reg/{patient_path}_Atlas_{atlas_name}_InSubjectDWISpaceFrom_AP.nii.gz"

    brain_mask, _ = load_nifti(brain_mask_path)

    brain_mask_mask = np.where(np.isin(brain_mask, atlas_values), 0, 1)

    b_values_x = [1000, 3000] if method_name_x == "DTI" else [1000, 3000, 5000, 10000]
    b_values_y = [1000, 3000] if method_name_y == "DTI" else [1000, 3000, 5000, 10000]
    if method_name_x == "MF" or method_name_y == "MF":
        b_values_x = b_values_y = [1000, 3000, 5000]  # Assuming 'MF' method uses these b-values

    # Determine the common b-values to process
    b_values = list(set(b_values_x).intersection(b_values_y))

    data_path_x = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/{method_name_x}/"

    data_path_y = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/{method_name_y}/"

    if method_name_x == "noddi":
        data_x, _ = load_nifti(data_path_x + f"{patient_path}_{data_type_x}.nii.gz")
    elif method_name_x == "MF":
        data_x, _ = load_nifti(f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/{patient_path}_{data_type_x}.nii.gz")
    else:
        data_x, _ = load_nifti(data_path_x + f"{patient_path}_{data_type_x}.nii.gz")

    data_x = ma.masked_array(data_x, mask=brain_mask_mask)

    x_all = data_x.compressed()


    if method_name_y == "noddi":
        data_y, _ = load_nifti(data_path_y + f"{patient_path}_{data_type_y}.nii.gz")
    elif method_name_y == "MF":
        data_y, _ = load_nifti(f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/{patient_path}_{data_type_y}.nii.gz")
    else:
        data_y, _ = load_nifti(data_path_y + f"{patient_path}_{data_type_y}.nii.gz")

    data_y = ma.masked_array(data_y, mask=brain_mask_mask)

    y_all = data_y.compressed()

    print(b_values)
    for bval in b_values:
        print(bval)
        if method_name_x == "noddi":
            data_x, _ = load_nifti(data_path_x + f"{patient_path}_b{bval}_{data_type_x}.nii.gz")
        elif method_name_x == "MF":
            data_x, _ = load_nifti(f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/{patient_path}_b{bval}_{data_type_x}.nii.gz")
        else:
            data_x, _ = load_nifti(data_path_x + f"{patient_path}_{data_type_x}_b{bval}.nii.gz")

        data_x = ma.masked_array(data_x, mask=brain_mask_mask)

        x = data_x.compressed()


        if method_name_y == "noddi":
            data_y, _ = load_nifti(data_path_y + f"{patient_path}_b{bval}_{data_type_y}.nii.gz")
        elif method_name_y == "MF":
            data_y, _ = load_nifti(f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/{patient_path}_b{bval}_{data_type_y}.nii.gz")
        else:
            data_y, _ = load_nifti(data_path_y + f"{patient_path}_{data_type_y}_b{bval}.nii.gz")

        data_y = ma.masked_array(data_y, mask=brain_mask_mask)

        y = data_y.compressed()
        
   
        gs = gridspec.GridSpec(3, 3)
        ax_main = plt.subplot(gs[1:3, :2])
        ax_xDist = plt.subplot(gs[0, :2],sharex=ax_main)
        ax_yDist = plt.subplot(gs[1:3, 2],sharey=ax_main)
            
        ax_main.scatter(x, y, color='blue', marker='.', label=f"{bval}")
        ax_main.scatter(x_all, y_all, color='red', marker='.', label='All shell')
        ax_main.set(xlabel=data_type_x, ylabel=data_type_y)
        ax_main.legend() 

        ax_xDist.hist(x,bins=100,align='mid',color='blue')
        ax_xDist.hist(x_all,bins=100,align='mid',color='red')
        ax_xDist.set(ylabel='count')
        
        ax_yDist.hist(y,bins=100,orientation='horizontal',align='mid')
        ax_yDist.hist(y_all,bins=100,orientation='horizontal',align='mid',color='red')
        ax_yDist.set(xlabel='count')

        folder_path = os.path.abspath(os.path.dirname(os.getcwd()))

        save_path = folder_path + f"/images/{method_name_x}/scatter_comparison_all_{roi_name}_{patient_path}_{method_name_x}_{data_type_x}_{method_name_y}_{data_type_y}_b{bval}.png"
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        if save_as_png:
            plt.savefig(save_path)
            print(f"Plot saved as PNG: {save_path}")
        if show:
            plt.show()
        plt.close()
    return


#########################################################################################################################################
    

method_name_x ="noddi"
method_name_y ="MF"

data_type_x = "odi"
data_type_y = "fvf_f0"

atlas_name = "JHU-ICBM-labels-1mm"

atlas_values = [5]
roi_name ="body_of_corpus_callosum"


for i in range(1001,1011):
    patient_path = f"sub-{i}"
    comparison_scatter_hist(patient_path,atlas_name, atlas_values, roi_name, method_name_x,method_name_y, data_type_x,data_type_y, show=False, save_as_png=True)    
    comparison_with_allshell_scatter_hist(patient_path,atlas_name, atlas_values, roi_name, method_name_x,method_name_y, data_type_x,data_type_y, show=False, save_as_png=True)
