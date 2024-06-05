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

######################################################################### MAPS #########################################################################

def plot_values_bvals(patient_path, method_name):
    """
    Plots a grid of images for a given patient and list of characteristics,
    with image normalization between 0 and 1.
    """

    folder_path = os.path.dirname(os.getcwd())

    if method_name == "DTI":
        lst_bvals = [1000, 3000]
        lst_parameter = ["FA","AD","MD","RD"]

    elif method_name == "noddi":
        lst_bvals = [1000, 3000,5000,10000]
        lst_parameter = ["odi","icvf","fiso","fbundle","fextra","fintra"]

    elif method_name == "MF":
        lst_bvals = [1000, 3000,5000]
        lst_parameter = ["frac_f1","frac_f0","frac_csf","fvf_f1","fvf_f0","fvf_tot"]

    else:
        print("error")


    fig, axes = plt.subplots(
        nrows=len(lst_parameter),
        ncols= len(lst_bvals)+1,
        figsize=(10, 10),
        squeeze=False,
    )

    if method_name == "MF":
        data_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/"
    else:
        data_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/{method_name}/"
    

    for i,parameter in enumerate(lst_parameter):
        
        for j, bval in enumerate(lst_bvals):
            if method_name == "noddi" or method_name == "MF":
                data, _ = load_nifti(data_path + f"{patient_path}_b{bval}_{parameter}.nii.gz")
            else:
                data, _ = load_nifti(data_path + f"{patient_path}_{parameter}_b{bval}.nii.gz")

            im = axes[i,j].imshow(data[:,:,50],vmin=0, vmax=1)
            axes[i, j].set_xticks([])  # Remove x-axis ticks
            axes[i, j].set_yticks([])  # Remove y-axis ticks

            if i == 0:
                axes[i, j].set_title(f'bvalue: {bval}', fontsize=8)

            # Add row labels
            if j == 0:
                axes[i, j].set_ylabel(parameter, fontsize=14)

        data, _ = load_nifti(data_path + f"{patient_path}_{parameter}.nii.gz")

        im = axes[i,-1].imshow(data[:,:,50],vmin=0, vmax=1)
        if i == 0:
            axes[i, -1].set_title('all shells', fontsize=9)
        axes[i, -1].set_xticks([])  # Remove x-axis ticks
        axes[i, -1].set_yticks([])  # Remove y-axis ticks
        

    fig.subplots_adjust(wspace=0.1, hspace=0.1, right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)


    save_path = folder_path + f"/images/maps/{method_name}/{patient_path}_{method_name}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path,dpi=1200)
    print(f"Plot saved as PNG: {save_path}")
    plt.close()



    return


def plot_values_bvals_dti(patient_path, method_name):
    """
    Plots a grid of images for a given patient and list of characteristics,
    with image normalization between 0 and 1.
    """

    folder_path = os.path.dirname(os.getcwd())

    if method_name == "DTI":
        lst_bvals = [1000, 3000]
        lst_parameter = ["FA", "AD", "MD", "RD"]

    elif method_name == "noddi":
        lst_bvals = [1000, 3000, 5000, 10000]
        lst_parameter = ["odi", "icvf", "fiso", "fbundle", "fextra", "fintra"]

    elif method_name == "MF":
        lst_bvals = [1000, 3000, 5000]
        lst_parameter = ["frac_f1", "frac_f0", "frac_csf", "fvf_f1", "fvf_f0", "fvf_tot"]

    else:
        print("error")
        return

    fig, axes = plt.subplots(
        nrows=len(lst_parameter),
        ncols=len(lst_bvals) + 1,
        figsize=(10, 10),
        squeeze=False,
    )

    if method_name == "MF":
        data_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/"
    else:
        data_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/{method_name}/"

    for i, parameter in enumerate(lst_parameter):

        for j, bval in enumerate(lst_bvals):
            
            if method_name == "noddi" or method_name == "MF":
                data, _ = load_nifti(data_path + f"{patient_path}_b{bval}_{parameter}.nii.gz")
            else:
                data, _ = load_nifti(data_path + f"{patient_path}_{parameter}_b{bval}.nii.gz")

            im = axes[i, j].imshow(data[:, :, 50], vmin=0, vmax=np.max(data[:, :, 50]))
            axes[i, j].set_xticks([])  # Remove x-axis ticks
            axes[i, j].set_yticks([])  # Remove y-axis ticks

            if i == 0:
                axes[i, j].set_title(f'bvalue: {bval}', fontsize=8)

            # Add row labels
            if j == 0:
                axes[i, j].set_ylabel(parameter, fontsize=14)
                
            

        data, _ = load_nifti(data_path + f"{patient_path}_{parameter}.nii.gz")
        im = axes[i, -1].imshow(data[:, :, 50], vmin=0, vmax=np.max(data[:, :, 50]))
        if i == 0:
            axes[i, -1].set_title('all shells', fontsize=9)
        axes[i, -1].set_xticks([])  # Remove x-axis ticks
        axes[i, -1].set_yticks([])  # Remove y-axis ticks

        # Add a colorbar for each row
        cbar_ax = fig.add_axes([0.92, 0.15 + (3-i) * 0.75 / len(lst_parameter), 0.02, 0.6 / len(lst_parameter)])
        fig.colorbar(im, cax=cbar_ax)

    fig.subplots_adjust(wspace=0.1, hspace=0.1, right=0.85)

    save_path = folder_path + f"/images/maps/{method_name}/{patient_path}_{method_name}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=1200)
    print(f"Plot saved as PNG: {save_path}")
    plt.close()

    return


def plot_values_reduced(patient_path, method_name,bval_reduced):
    """
    Plots a grid of images for a given patient and list of characteristics,
    with image normalization between 0 and 1.
    """

    folder_path = os.path.dirname(os.getcwd())

    if method_name == "DTI":
        lst_parameter = ["FA","AD","MD","RD"]

    elif method_name == "noddi":
        lst_parameter = ["odi","icvf","fiso","fbundle","fextra","fintra"]

    elif method_name == "MF":
        lst_parameter = ["frac_f1","frac_f0","frac_csf","fvf_f1","fvf_f0","fvf_tot"]

    else:
        print("error")

    
    if bval_reduced == 1000:
        lst_direction =[16,32,40,48,64]

    elif bval_reduced == 3000:
        lst_direction =[16,32,40,48,64]
    
    elif bval_reduced == 5000:
        lst_direction =[32,64,100,128]

    elif bval_reduced == 10000:
        lst_direction =[128,200,252]


    fig, axes = plt.subplots(
        nrows=len(lst_parameter),
        ncols= len(lst_direction),
        figsize=(10, 10),
        squeeze=False,
    )

    if method_name == "MF":
        data_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/"
    else:
        data_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/{method_name}/"
    

    for i,parameter in enumerate(lst_parameter):
        
        for j, direction in enumerate(lst_direction[:-1]):

            data, _ = load_nifti(data_path + f"{patient_path}_reduced_b{bval_reduced}_{direction}_{parameter}.nii.gz")

            im = axes[i,j].imshow(data[:,:,50],vmin=0, vmax=1)
            axes[i, j].set_xticks([])  # Remove x-axis ticks
            axes[i, j].set_yticks([])  # Remove y-axis ticks

            if i == 0:
                axes[i, j].set_title(f'{direction}', fontsize=8)

            # Add row labels
            if j == 0:
                axes[i, j].set_ylabel(parameter, fontsize=14)

        data, _ = load_nifti(data_path + f"{patient_path}_{parameter}.nii.gz")

        im = axes[i,-1].imshow(data[:,:,50],vmin=0, vmax=1)
        if i == 0:
            axes[i, -1].set_title(f'{lst_direction[-1]}', fontsize=9)
        axes[i, -1].set_xticks([])  # Remove x-axis ticks
        axes[i, -1].set_yticks([])  # Remove y-axis ticks
        

    fig.subplots_adjust(wspace=0.1, hspace=0.1, right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)


    save_path = folder_path + f"/images/maps/{method_name}/{patient_path}_{method_name}_b{bval_reduced}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path,dpi=1200)
    print(f"Plot saved as PNG: {save_path}")
    plt.close()


    return


def plot_values_reduced_DTI(patient_path, method_name,bval_reduced):
    """
    Plots a grid of images for a given patient and list of characteristics,
    with image normalization between 0 and 1.
    """

    folder_path = os.path.dirname(os.getcwd())

    if method_name == "DTI":
        lst_parameter = ["FA","AD","MD","RD"]

    elif method_name == "noddi":
        lst_parameter = ["odi","icvf","fiso","fbundle","fextra","fintra"]

    elif method_name == "MF":
        lst_parameter = ["frac_f1","frac_f0","frac_csf","fvf_f1","fvf_f0","fvf_tot"]

    else:
        print("error")

    
    if bval_reduced == 1000:
        lst_direction =[16,32,40,48,64]

    elif bval_reduced == 3000:
        lst_direction =[16,32,40,48,64]
    
    elif bval_reduced == 5000:
        lst_direction =[32,64,100,128]

    elif bval_reduced == 10000:
        lst_direction =[128,200,252]


    fig, axes = plt.subplots(
        nrows=len(lst_parameter),
        ncols= len(lst_direction),
        figsize=(10, 10),
        squeeze=False,
    )

    if method_name == "MF":
        data_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/"
    else:
        data_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/{method_name}/"
    

    for i,parameter in enumerate(lst_parameter):
        
        for j, direction in enumerate(lst_direction[:-1]):

            data, _ = load_nifti(data_path + f"{patient_path}_reduced_b{bval_reduced}_{direction}_{parameter}.nii.gz")

            im = axes[i,j].imshow(data[:,:,50],vmin=0, vmax=np.max(data[:,:,50]))
            axes[i, j].set_xticks([])  # Remove x-axis ticks
            axes[i, j].set_yticks([])  # Remove y-axis ticks

            if i == 0:
                axes[i, j].set_title(f'{direction}', fontsize=8)

            # Add row labels
            if j == 0:
                axes[i, j].set_ylabel(parameter, fontsize=14)

        data, _ = load_nifti(data_path + f"{patient_path}_{parameter}.nii.gz")

        im = axes[i,-1].imshow(data[:,:,50],vmin=0, vmax=np.max(data[:,:,50]))
        if i == 0:
            axes[i, -1].set_title(f'{lst_direction[-1]}', fontsize=9)
        axes[i, -1].set_xticks([])  # Remove x-axis ticks
        axes[i, -1].set_yticks([])  # Remove y-axis ticks

        cbar_ax = fig.add_axes([0.92, 0.15 + (3-i) * 0.75 / len(lst_parameter), 0.02, 0.6 / len(lst_parameter)])
        fig.colorbar(im, cax=cbar_ax)
        

    fig.subplots_adjust(wspace=0.1, hspace=0.1, right=0.8)


    save_path = folder_path + f"/images/maps/{method_name}/{patient_path}_{method_name}_b{bval_reduced}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path,dpi=1200)
    print(f"Plot saved as PNG: {save_path}")
    plt.close()

    return 

def plot_values_reduced_ROI(patient_path, method_name, bval_reduced, atlas_values, atlas_name, roi_name):
    """
    Plots a grid of images for a given patient and list of characteristics,
    with image normalization between 0 and 1.
    """
    
    print(f"Starting plot generation for patient: {patient_path}, method: {method_name}, bval_reduced: {bval_reduced}, ROI: {roi_name}")
    
    folder_path = os.path.dirname(os.getcwd())
    subject_folder = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}"

    # Load the brain mask for excluding non-brain areas from analysis
    brain_mask_path = f"{subject_folder}/reg/{patient_path}_Atlas_{atlas_name}_InSubjectDWISpaceFrom_AP.nii.gz"
    brain_mask, _ = load_nifti(brain_mask_path)
    brain_mask = np.where(np.isin(brain_mask, atlas_values), 0, 1)
    print("Brain mask loaded and processed.")

    if method_name == "DTI":
        lst_parameter = ["FA", "AD", "MD", "RD"]
    elif method_name == "noddi":
        lst_parameter = ["odi", "icvf", "fiso", "fbundle", "fextra", "fintra"]
    elif method_name == "MF":
        lst_parameter = ["frac_f1", "frac_f0", "frac_csf", "fvf_f1", "fvf_f0", "fvf_tot"]
    else:
        print("Error: Unsupported method name.")
        return

    if bval_reduced == 1000:
        lst_direction = [16, 32, 40, 48, 64]
    elif bval_reduced == 3000:
        lst_direction = [16, 32, 40, 48, 64]
    elif bval_reduced == 5000:
        lst_direction = [32, 64, 100, 128]
    elif bval_reduced == 10000:
        lst_direction = [128, 200, 252]
    else:
        print("Error: Unsupported bval_reduced value.")
        return

    fig, axes = plt.subplots(
        nrows=len(lst_parameter),
        ncols=len(lst_direction),
        figsize=(5, 10),
        squeeze=False,
    )
    print("Figure and axes created.")

    if method_name == "MF":
        data_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/"
    else:
        data_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/{method_name}/"

    for i, parameter in enumerate(lst_parameter):
        print(f"Processing parameter: {parameter}")
        
        for j, direction in enumerate(lst_direction[:-1]):
            print(f"Processing direction: {direction}")
            
            data, _ = load_nifti(data_path + f"{patient_path}_reduced_b{bval_reduced}_{direction}_{parameter}.nii.gz")
            data[brain_mask==1] = 0
            im = axes[i, j].imshow(data[:, :, 50], vmin=0, vmax=1)
            axes[i, j].set_xticks([])  # Remove x-axis ticks
            axes[i, j].set_yticks([])  # Remove y-axis ticks

            if i == 0:
                axes[i, j].set_title(f'{direction}', fontsize=8)

            if j == 0:
                axes[i, j].set_ylabel(parameter, fontsize=14)

        data, _ = load_nifti(data_path + f"{patient_path}_{parameter}.nii.gz")
        data[brain_mask==1] = 0
        im = axes[i, -1].imshow(data[:, :, 50], vmin=0, vmax=1)
        if i == 0:
            axes[i, -1].set_title(f'{lst_direction[-1]}', fontsize=9)
        axes[i, -1].set_xticks([])  # Remove x-axis ticks
        axes[i, -1].set_yticks([])  # Remove y-axis ticks

    fig.subplots_adjust(wspace=0.1, hspace=0.1, right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    print("Colorbar added to the figure.")

    save_path = folder_path + f"/images/maps/{method_name}/{patient_path}_{method_name}_b{bval_reduced}_{roi_name}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=1200)
    print(f"Plot saved as PNG: {save_path}")
    plt.close()
    print("Figure closed.")

    return


def plot_values_pair(patient_path, method_name):
    """
    Plots a grid of images for a given patient and list of characteristics,
    with image normalization between 0 and 1.
    """

    folder_path = os.path.dirname(os.getcwd())

    if method_name == "DTI":
        lst_bvals = [1000, 3000]
        lst_parameter = ["FA","AD","MD","RD"]

    elif method_name == "noddi":
        lst_bvals = [1000, 3000,5000,10000]
        lst_parameter = ["odi","icvf","fiso","fbundle","fextra","fintra"]

    elif method_name == "MF":
        lst_bvals = [1000, 3000,5000,10000]
        lst_parameter = ["frac_f1","frac_f0","frac_csf","fvf_f1","fvf_f0","fvf_tot"]

    else:
        print("error")


    bval_combinations = [(b1, b2) for i, b1 in enumerate(lst_bvals) for b2 in lst_bvals[i+1:] if b1 != b2]


    fig, axes = plt.subplots(
        nrows=len(lst_parameter),
        ncols= len(bval_combinations)+1,
        figsize=(8, 8),
        squeeze=False,
    )

    if method_name == "MF":
        data_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/"
    else:
        data_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/{method_name}/"
    

    for i,parameter in enumerate(lst_parameter):
        
        for j, bval_pair in enumerate(bval_combinations):

            bval_str = f"b{bval_pair[0]}b{bval_pair[1]}"
            
            if method_name == "noddi" or method_name == "MF":
                data, _ = load_nifti(data_path + f"{patient_path}_{bval_str}_{parameter}.nii.gz")
            else:
                data, _ = load_nifti(data_path + f"{patient_path}_{parameter}_{bval_str}.nii.gz")

            im = axes[i,j].imshow(data[:,:,50],vmin=0, vmax=1)
            axes[i, j].set_xticks([])  # Remove x-axis ticks
            axes[i, j].set_yticks([])  # Remove y-axis ticks

            if i == 0:
                axes[i, j].set_title(f"b{bval_pair[0]}+b{bval_pair[1]}", fontsize=5)

            # Add row labels
            if j == 0:
                axes[i, j].set_ylabel(parameter, fontsize=14)

        data, _ = load_nifti(data_path + f"{patient_path}_{parameter}.nii.gz")

        im = axes[i,-1].imshow(data[:,:,50],vmin=0, vmax=1)
        if i == 0:
            axes[i, -1].set_title('all shells', fontsize=7)
        axes[i, -1].set_xticks([])  # Remove x-axis ticks
        axes[i, -1].set_yticks([])  # Remove y-axis ticks
        

    fig.subplots_adjust(wspace=0.1, hspace=0.1, right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)


    save_path = folder_path + f"/images/maps/{method_name}/{patient_path}_{method_name}_pair.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=1200)
    print(f"Plot saved as PNG: {save_path}")
    plt.close()



    return

def plot_values_pair_DTI(patient_path, method_name):
    """
    Plots a grid of images for a given patient and list of characteristics,
    with image normalization between 0 and 1.
    """

    folder_path = os.path.dirname(os.getcwd())

    if method_name == "DTI":
        lst_bvals = [1000, 3000]
        lst_parameter = ["FA","AD","MD","RD"]

    elif method_name == "noddi":
        lst_bvals = [1000, 3000,5000,10000]
        lst_parameter = ["odi","icvf","fiso","fbundle","fextra","fintra"]

    elif method_name == "MF":
        lst_bvals = [1000, 3000,5000,10000]
        lst_parameter = ["frac_f1","frac_f0","frac_csf","fvf_f1","fvf_f0","fvf_tot"]

    else:
        print("error")


    bval_combinations = [(b1, b2) for i, b1 in enumerate(lst_bvals) for b2 in lst_bvals[i+1:] if b1 != b2]


    fig, axes = plt.subplots(
        nrows=len(lst_parameter),
        ncols= len(bval_combinations)+1,
        figsize=(10, 10),
        squeeze=False,
    )

    if method_name == "MF":
        data_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/"
    else:
        data_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/{method_name}/"
    

    for i,parameter in enumerate(lst_parameter):
        
        for j, bval_pair in enumerate(bval_combinations):

            bval_str = f"b{bval_pair[0]}b{bval_pair[1]}"
            
            if method_name == "noddi" or method_name == "MF":
                data, _ = load_nifti(data_path + f"{patient_path}_{bval_str}_{parameter}.nii.gz")
            else:
                data, _ = load_nifti(data_path + f"{patient_path}_{parameter}_{bval_str}.nii.gz")

            im = axes[i,j].imshow(data[:,:,50],vmin=0, vmax=np.max(data[:,:,50]))
            axes[i, j].set_xticks([])  # Remove x-axis ticks
            axes[i, j].set_yticks([])  # Remove y-axis ticks

            if i == 0:
                axes[i, j].set_title(f"b{bval_pair[0]}+b{bval_pair[1]}", fontsize=5)

            # Add row labels
            if j == 0:
                axes[i, j].set_ylabel(parameter, fontsize=14)

        data, _ = load_nifti(data_path + f"{patient_path}_{parameter}.nii.gz")

        im = axes[i,-1].imshow(data[:,:,50],vmin=0, vmax=np.max(data[:,:,50]))
        if i == 0:
            axes[i, -1].set_title('all shells', fontsize=7)
        axes[i, -1].set_xticks([])  # Remove x-axis ticks
        axes[i, -1].set_yticks([])  # Remove y-axis ticks

        cbar_ax = fig.add_axes([0.92, 0.15 + (3-i) * 0.75 / len(lst_parameter), 0.02, 0.6 / len(lst_parameter)])
        fig.colorbar(im, cax=cbar_ax)
        

    fig.subplots_adjust(wspace=0.1, hspace=0.1, right=0.8)



    save_path = folder_path + f"/images/maps/{method_name}/{patient_path}_{method_name}_pair.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=1200)
    print(f"Plot saved as PNG: {save_path}")
    plt.close()



    return
######################################################################### Parameter ######################################################
def plot_slices_boxplot_roi(patient_path, atlas_name, atlas_values, roi_name, method_name, data_type, preprocessing_type, show=False, save_as_png=False):
    """
    Plots slices from different b-values and compares statistical distributions within regions of interest (ROI) for MRI data. 
    This function overlays the selected MRI data slices over T1-weighted images to provide anatomical context and plots the 
    normalized histograms of the pixel values within the defined ROI for each b-value. This visualization aids in assessing 
    the impact of different MRI processing methods and preprocessing types within specific anatomical regions.

    Parameters:
        patient_path (str): Path to the patient-specific data directory.
        atlas_name (str): Name of the atlas used for ROI segmentation.
        atlas_values (list of int): List of atlas-specific integers representing the ROI.
        roi_name (str): Descriptive name of the ROI.
        method_name (str): Name of the MRI processing method applied (e.g., 'DTI', 'noddi', 'MF').
        data_type (str): Type of MRI data being visualized (e.g., 'FA', 'MD').
        preprocessing_type (str): Type of preprocessing applied to the MRI data.
        show (bool, optional): If True, the plot will be displayed. Defaults to False.
        save_as_png (bool, optional): If True, the plot will be saved as a PNG file. Defaults to False.

    Returns:
        None
    """
    folder_path = os.path.abspath(os.path.dirname(os.getcwd()))

    # Construct the file paths for the patient's data
    subject_folder = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}"
    T1_subject = f'{subject_folder}/reg/{patient_path}_T1_DWIspace.nii.gz'
    
    # Load the T1-weighted brain image for anatomical reference
    T1_subject_data, _ = load_nifti(T1_subject)
    print(f"T1 image path set to: {T1_subject}")


    # Load the brain mask for excluding non-brain areas from analysis
    brain_mask_path = f"{subject_folder}/reg/{patient_path}_Atlas_{atlas_name}_InSubjectDWISpaceFrom_AP.nii.gz"
    brain_mask, _ = load_nifti(brain_mask_path)
    # Create a binary mask based on the specified ROI values
    brain_mask_mask = np.where(np.isin(brain_mask, atlas_values), 0, 1)


    print(f"Shape mask mask is {np.shape(brain_mask_mask)}")


    masked = np.ma.masked_where(brain_mask_mask == 0, brain_mask_mask)

    # Define the base path for the processed MRI data
    data_path = f"{subject_folder}/dMRI/{method_name}/"

    # Set different b-values based on the MRI method
    if method_name == "DTI":
        b_values = [1000, 3000]
    else:
        b_values = [1000, 3000, 5000, 10000]

    # Initialize lists to hold the filtered data and selected slices
    filtered_data = []
    slice_selected = []

    # Specify the slice index to be visualized
    slice_to_take = 50  # Example slice index, can be changed based on the dataset

    # Load and process the data for each b-value
    for b_value in b_values:
        # Determine the filename based on the processing method and b-value
        filename = f"{patient_path}_b{b_value}_{data_type}.nii.gz" if method_name == "noddi" else f"{patient_path}_{data_type}_b{b_value}.nii.gz"
        data, _ = load_nifti(data_path + filename)
        print(f"Loading data from: {data_path + filename}")

        # Store the selected slice and apply the brain mask
        slice_selected.append(data[:,:,slice_to_take])
        mx = ma.masked_array(data, mask=brain_mask_mask)
        filtered_data.append(mx.compressed())

    # Additional processing if method is 'MF'
    if method_name == "MF":
        data, _ = load_nifti(f"{subject_folder}/dMRI/microstructure/mf/{patient_path}_{data_type}.nii.gz")
        slice_selected.append(data[:,:,slice_to_take])
    else:   
        data, _ = load_nifti(data_path + f"{patient_path}_{data_type}.nii.gz")
        slice_selected.append(data[:,:,slice_to_take])
    
    print(f"Shape data is {np.shape(data)}")

    # Mask the data and append to filtered data for statistical analysis
    mx = ma.masked_array(data, mask=brain_mask_mask)
    filtered_data.append(mx.compressed())

    # Plot the T1 image and the segmented ROI on top of it
    num_b_values = len(b_values)
    fig, axes = plt.subplots(1, num_b_values + 3, figsize=(5 * (num_b_values + 3), 5))
    # Display the T1-weighted anatomical image in grayscale
    axes[0].imshow(T1_subject_data[:, :, slice_to_take], cmap='gray', interpolation='none')

    # Create a mask where only the zones of value 255 in brain_segmentation_segmentation are True
    overlay_mask = masked[:, :, slice_to_take]

    # Overlay the segmentation map on top of the T1 image using the mask
    # 'jet' is the colormap used for the segmentation overlay
    # alpha < 1 makes the overlay semi-transparent so the T1 image can be seen underneath
    # Note: Use the overlay_mask to selectively display the segmentation
    axes[0].imshow(overlay_mask, cmap='jet', interpolation='none', alpha=0.7)


    # Plot each selected slice for the different b-values
    for i, slice_data in enumerate(slice_selected[:-1]):  # Excluding the last added slice if 'MF' is used
        axes[i+1].imshow(slice_data, cmap='gray')
        axes[i+1].set_title(f'{method_name} - B{b_values[i]} - {data_type} Slice')

    # Plot the combined data slice if method is 'MF'
    axes[num_b_values + 1].imshow(slice_selected[-1], cmap='gray')
    axes[num_b_values + 1].set_title(f'{method_name} - All bvalues - {data_type} Slice')

    # Plot the normalized histograms for the pixel values within the ROI for each b-value
    axes[-1].boxplot(filtered_data, labels=[f'B{b_values[i]}' for i in range(num_b_values)]+["all bvals"],showfliers=False)
    axes[-1].legend()

    print("Histograms plotted successfully.")

    # Display or save the plot according to the user's request
    if show:
        plt.show()
    
    if save_as_png:
        save_path = folder_path + f"/images/{method_name}/slices_and_histogram_b_values_comparison_roi_{roi_name}_{patient_path}_{method_name}_{data_type}_{preprocessing_type}.png"
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path)
        print(f"Plot saved as PNG: {save_path}")

    plt.close()

    return 

######################################################################### MSE ######################################################
  
def calculate_and_plot_comparison_reduced_boxplot_mse(patient_path,reduced_bval, reduced_nb, method_name, show=False, save_as_png=False):
    """
    Visualizes the comparison of Mean Squared Error (MSE) between original and reduced datasets for multiple b-values.

    Parameters:
    - patient_path (str): The name of the patient.
    - reduced_bval (int): Specific b-value or list of b-values that have been reduced.
    - reduced_nb (int or list): Number of directions or specific reductions to compare.
    - method_name (str): The imaging method (e.g., "noddi", "mf", "dti").
    - show (bool): If True, displays the plot. Default is False.
    - save_as_png (bool): If True, saves the plot as a PNG file. Default is False.

    Returns:
    - None
    """
    folder_path = os.path.dirname(os.getcwd())
    images_path = os.path.join(folder_path, "images", "mse", method_name)

    method_path = os.path.join("/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects", patient_path, "dMRI", method_name)
    brain_mask_path =  f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/masks/{patient_path}_brain_mask.nii.gz"

    brain_mask, _ = load_nifti(brain_mask_path)
    brain_mask = brain_mask == 1

    # Load original MSE data
    mse_original_path = os.path.join(method_path, f"{patient_path}_mse.nii.gz")
    mse_original, _ = load_nifti(mse_original_path)
    mse_original = mse_original[brain_mask].flatten()
    mse_original = mse_original[~np.isnan(mse_original)]

    # Initialize data for boxplot
    mse_data = [mse_original]
    labels = ['Original']

    # Load and add reduced MSE data
    if isinstance(reduced_nb, list):
        for k in reduced_nb:
            mse_reduced_path = os.path.join(method_path, f"{patient_path}_reduced_b{reduced_bval}_{k}_mse.nii.gz")
            mse_reduced, _ = load_nifti(mse_reduced_path)
            mse_reduced = mse_reduced[brain_mask].flatten()
            mse_reduced = mse_reduced[~np.isnan(mse_reduced)]
            mse_data.append(mse_reduced)
            labels.append(f"Reduced {k}")
    else:
        mse_reduced_path = os.path.join(method_path, f"{patient_path}_reduced_b{reduced_bval}_{reduced_nb}_mse.nii.gz")
        mse_reduced, _ = load_nifti(mse_reduced_path)
        mse_reduced = mse_reduced[brain_mask].flatten()
        mse_reduced = mse_reduced[~np.isnan(mse_reduced)]
        mse_data.append(mse_reduced)
        labels.append(f"Reduced {reduced_nb}")

    # Plot comparison boxplot of MSE
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=mse_data, orient='vertical', width=0.3, showfliers=False)
    plt.xticks(range(len(labels)), labels,fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(f"MSE Comparison for {method_name.upper()} - B{reduced_bval}, Reductions: {reduced_nb}, Patient: {patient_path}")

    # Save or show the plot
    save_plot_path = os.path.join(images_path, f"comparison_reduced_b{reduced_bval}_{reduced_nb}_boxplot_mse_{patient_path}.png")
    os.makedirs(os.path.dirname(save_plot_path), exist_ok=True)
    if save_as_png:
        plt.savefig(save_plot_path)
    if show:
        plt.show()
    plt.close()


def calculate_and_plot_hist_mse_multi_methods(patient_path, method_list, preproc_raw="preproc", show=False, save_as_png=False):
  """
  Calculates and plots the histogram of Mean Squared Error (MSE) for multiple methods.

  Args:
      patient_path (str): Path to the patient's data directory.
      method_list (list): List of method names (e.g., ["noddi", "mf", "dti"]).
      preproc_raw (str, optional): Preprocessing stage directory name. Defaults to "preproc".
      show (bool, optional): Whether to display the plot using plt.show(). Defaults to True.
      save_as_png (bool, optional): Whether to save the plot as a PNG file. Defaults to False.

  Returns:
      None
  """
  folder_path = os.path.dirname(os.getcwd())

  images_path = folder_path + "/images/mse/"

  brain_mask_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/masks/{patient_path}_brain_mask.nii.gz"
  #brain_mask_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/masks/{patient_path}_wm_mask_AP.nii.gz"

  brain_mask, _ = load_nifti(brain_mask_path)

  brain_mask_mask = brain_mask == 1


  plt.figure(figsize=(12, 6))

  for method_name in method_list:
    if method_name == "MF":
        method_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/{patient_path}_MSE.nii.gz"
        mse, _ = load_nifti(method_path)
    else:
        method_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/{method_name}/"
        mse, _ = load_nifti(method_path + f"{patient_path}_mse.nii.gz")

    mse = mse[brain_mask_mask]

    mse = mse.flatten()
    mse = mse[np.logical_not(np.isnan(mse))]

    # Calculate statistics
    mean_mse = np.nanmean(mse)
    std_dev_mse = np.nanstd(mse)
    min_mse = np.nanmin(mse)
    max_mse = np.nanmax(mse)

    # Print statistics
    print(f"Statistical Information about MSE ({method_name.upper()} - {patient_path}):")
    print(f" - Mean: {mean_mse:.4f}")
    print(f" - Median: {np.nanmedian(mse):.4f}")
    print(f" - Standard Deviation: {std_dev_mse:.4f}")
    print(f" - Minimum: {min_mse}")
    print(f" - Maximum: {max_mse}")

    # Histogram with informative labels and error handling
    hist_range = (0, mean_mse + 3 * std_dev_mse)
    plt.hist(mse, bins=100, range=hist_range, label=method_name.upper())  # Adjust bins as needed

  plt.xlabel("MSE")
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  plt.ylabel("Frequency")
  plt.title(f"Histogram MSE - {patient_path}")
  plt.legend()

  # Save plot with path handling
  save_plot_path = os.path.join(images_path, f"hist_mse_{patient_path}.png")
  if not os.path.exists(os.path.dirname(save_plot_path)):
            os.makedirs(os.path.dirname(save_plot_path))

  if save_as_png:
    plt.savefig(save_plot_path)

  if show:
    plt.show()

  plt.close()  # Explicitly close the plot to avoid memory issues

  return None


def calculate_and_plot_boxplot_mse_multi_methods_subplot(patient_path, method_list, prepoc_raw="preproc", show=True, save_as_png=False):
    """
    Calculate and plot the boxplot of Mean Squared Error (MSE) for multiple methods in subplots.

    Parameters:
    - patient_path (str): The name of the patient.
    - method_list (list): List of method names (e.g., ["noddi", "mf", "dti"]).
    - show (bool): Whether to display the plot using plt.show(). Default is True.
    - save_as_png (bool): Whether to save the plot as a PNG file. Default is False.

    Returns:
    - None
    """
    folder_path = os.path.dirname(os.getcwd())

    images_path = folder_path + "/images/mse/"

    brain_mask_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/masks/{patient_path}_brain_mask.nii.gz"

    brain_mask, _ = load_nifti(brain_mask_path)

    brain_mask_mask = brain_mask == 1

    

    num_methods = len(method_list)
    fig, axes = plt.subplots(1, num_methods, figsize=(4 * num_methods, 6))

    for i, method_name in enumerate(method_list):
        if method_name == "MF":
            method_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/{patient_path}_MSE.nii.gz"
            mse, _ = load_nifti(method_path)
        else:
            method_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/{method_name}/"
            mse, _ = load_nifti(method_path + f"{patient_path}_mse.nii.gz")

        mse = mse[brain_mask_mask]

        mse = mse.flatten()

        mse = mse[np.logical_not(np.isnan(mse))]

        # Calculate statistical information about MSE
        mean_mse = np.nanmean(mse)
        std_dev_mse = np.nanstd(mse)
        min_mse = np.nanmin(mse)
        max_mse = np.nanmax(mse)

        # Display statistical information about MSE in the terminal
        print(f"Statistical Information about MSE ({method_name.upper()} - {patient_path}):")
        print(f" - Mean: {mean_mse:.4f}")
        print(f" - Median: {np.nanmedian(mse):.4f}")
        print(f" - Standard Deviation: {std_dev_mse:.4f}")
        print(f" - Minimum: {min_mse}")
        print(f" - Maximum: {max_mse}")
        print()

        # Display statistical information about MSE
        stat_info = (
            f"Mean: {mean_mse:.4f}\n"
            f"Median: {np.nanmedian(mse):.4f}\n"
            f"Standard Deviation: {std_dev_mse:.4f}\n"
            f"Minimum: {min_mse}\n"
            f"Maximum: {max_mse}"
        )

        # Plot boxplot of MSE
        axes[i].boxplot(mse,showfliers=False, labels=[method_name.upper()], vert=True)
        axes[i].set_title(f"{method_name.upper()} - {patient_path}")

        # Add statistical information below the subplot
        axes[i].text(0.5, -0.25, stat_info, ha='center', va='center', transform=axes[i].transAxes, bbox=dict(facecolor='white', alpha=0.5))

    plt.suptitle(f"Boxplot MSE - {patient_path}")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()

    save_plot_path = f"{images_path}boxplot_mse_subplot_{patient_path}.png"

    if not os.path.exists(os.path.dirname(save_plot_path)):
            os.makedirs(os.path.dirname(save_plot_path))

    if save_as_png:
        plt.savefig(save_plot_path, bbox_inches='tight')

    if show:
        plt.show()

    return


def calculate_and_plot_boxplot_mse_multi_methods_1_fig(patient_path, method_list, prepoc_raw="preproc", show=False, save_as_png=False):
    """
    Calculate and plot the boxplot of Mean Squared Error (MSE) for multiple methods in subplots.

    Parameters:
    - patient_path (str): The name of the patient.
    - method_list (list): List of method names (e.g., ["noddi", "mf", "dti"]).
    - prepoc_raw (str): Specify whether to use preprocessed ("preproc") or raw ("raw") data.
    - show (bool): Whether to display the plot using plt.show(). Default is True.
    - save_as_png (bool): Whether to save the plot as a PNG file. Default is False.

    Returns:
    - None
    """
    folder_path = os.path.dirname(os.getcwd())
    images_path = folder_path + "/images/mse/"

    brain_mask_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/masks/{patient_path}_brain_mask.nii.gz"
    brain_mask, _ = load_nifti(brain_mask_path)
    brain_mask_mask = brain_mask == 1

    num_methods = len(method_list)
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, method_name in enumerate(method_list):
        if method_name == "MF":
            method_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/{patient_path}_MSE.nii.gz"
            mse, _ = load_nifti(method_path)
        else:
            method_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/{method_name}/"
            mse, _ = load_nifti(method_path + f"{patient_path}_mse.nii.gz")

        mse = mse[brain_mask_mask]
        mse = mse.flatten()
        mse = mse[np.logical_not(np.isnan(mse))]

        # Calculate statistical information about MSE
        mean_mse = np.nanmean(mse)
        std_dev_mse = np.nanstd(mse)
        min_mse = np.nanmin(mse)
        max_mse = np.nanmax(mse)

        # Display statistical information about MSE
        print(f"Statistical Information about MSE ({method_name.upper()} - {patient_path}):")
        print(f" - Mean: {mean_mse:.4f}")
        print(f" - Median: {np.nanmedian(mse):.4f}")
        print(f" - Standard Deviation: {std_dev_mse:.4f}")
        print(f" - Minimum: {min_mse}")
        print(f" - Maximum: {max_mse}")
        print()

        # Display statistical information about MSE
        stat_info = (
            f"Mean: {mean_mse:.4f}\n"
            f"Median: {np.nanmedian(mse):.4f}\n"
            f"Standard Deviation: {std_dev_mse:.4f}\n"
            f"Minimum: {min_mse}\n"
            f"Maximum: {max_mse}"
        )

        # Plot boxplot of MSE
        ax.boxplot(mse, positions=[i], showfliers=False, labels=[method_name.upper()], vert=True)

        # Add statistical information below the subplot
        #ax.text(i, -0.15, stat_info, ha='center', va='center', fontsize=8, bbox=dict(facecolor='white', alpha=0.5))

    ax.set_xticks(np.arange(num_methods))
    ax.set_xticklabels(method_list)
    ax.set_xlabel("Method")
    ax.set_ylabel("Mean Squared Error (MSE)")
    ax.set_title(f"Boxplot MSE - {patient_path}")
    ax.grid(True)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    save_plot_path = f"{images_path}boxplot_mse_subplot_{patient_path}_1_fig.png"

    if not os.path.exists(os.path.dirname(save_plot_path)):
            os.makedirs(os.path.dirname(save_plot_path))

    if save_as_png:
        plt.savefig(save_plot_path, bbox_inches='tight')

    if show:
        plt.show()
    
    plt.close()

    return


def visualize_method_MSE_by_bvalue(patient_path, method_name, prepoc_raw="preproc", show=False, save_as_png=False):
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

    images_path = folder_path + "/images/mse/" + method_name + "/"

    if method_name == "MF":
        method_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/"
    else:
        method_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/{method_name}/"

    brain_mask_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/masks/{patient_path}_brain_mask.nii.gz"

    brain_mask, _ = load_nifti(brain_mask_path)

    brain_mask_mask = brain_mask == 1

    if method_name == "DTI":
        b_values = [1000, 3000]
    else:
        b_values = [1000, 3000, 5000, 10000]

    mse_data = []

    for target_bvalue in b_values:

        if method_name == "DTI":
            mse, _ = load_nifti(method_path + f"{patient_path}_mse_b{target_bvalue}.nii.gz")
        
        elif method_name == "MF":
            print("MF")
            mse, _ = load_nifti(method_path+f"{patient_path}_b{target_bvalue}.0_MSE.nii.gz")
        else:
            mse, _ = load_nifti(method_path + f"{patient_path}_b{target_bvalue}_mse.nii.gz")

        mse = mse[brain_mask_mask]
        mse = mse.flatten()
        mse = mse[np.logical_not(np.isnan(mse))]
        mse_data.append(mse)

        # Calculate statistical information about MSE
        mean_mse = np.nanmean(mse)
        std_dev_mse = np.nanstd(mse)
        min_mse = np.nanmin(mse)
        max_mse = np.nanmax(mse)

        # Display statistical information about MSE in the terminal
        print(f"Statistical Information about MSE ({method_name.upper()} - {patient_path} - B{target_bvalue}):")
        print(f" - Mean: {mean_mse:.4f}")
        print(f" - Median: {np.nanmedian(mse):.4f}")
        print(f" - Standard Deviation: {std_dev_mse:.4f}")
        print(f" - Minimum: {min_mse}")
        print(f" - Maximum: {max_mse}")
        print()

    if method_name != "MF":
        mse, _ = load_nifti(method_path + f"{patient_path}_mse.nii.gz")

    else:
        mse, _ = load_nifti(method_path+f"{patient_path}_MSE.nii.gz")
    
    mse = mse[brain_mask_mask]
    mse = mse.flatten()
    mse = mse[np.logical_not(np.isnan(mse))]
    mse_data.append(mse)

    labels=[f"B{value}" for value in b_values]
    labels.append("all bvalues")

    # Plot boxplot of MSE for all b-values
    plt.figure(figsize=(8, 6))
    plt.boxplot(mse_data, showfliers=False, labels=labels, vert=True)
    plt.title(f"Boxplot MSE - {method_name.upper()} - {patient_path}")
    plt.xlabel("B-value")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    save_plot_path = f"{images_path}boxplot_mse_{method_name}_{patient_path}.png"

    if not os.path.exists(os.path.dirname(save_plot_path)):
            os.makedirs(os.path.dirname(save_plot_path))

    if save_as_png:
        plt.savefig(save_plot_path, bbox_inches='tight')

    if show:
        plt.show()

    return


def visualize_method_MSE_by_bvalue_pair(patient_path, method_name, prepoc_raw="preproc", show=False, save_as_png=False):
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

    images_path = folder_path + "/images/mse/" + method_name + "/"

    if method_name == "MF":
        method_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/"
    else:
        method_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/{method_name}/"

    brain_mask_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/masks/{patient_path}_brain_mask.nii.gz"

    brain_mask, _ = load_nifti(brain_mask_path)

    brain_mask_mask = brain_mask == 1

    if method_name == "DTI":
        b_values = [1000, 3000]
    else:
        b_values = [1000, 3000, 5000, 10000]

    bval_combinations = [(b1, b2) for i, b1 in enumerate(b_values) for b2 in b_values[i+1:] if b1 != b2]

    mse_data = []

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

        mse = mse[brain_mask_mask]
        mse = mse.flatten()
        mse = mse[np.logical_not(np.isnan(mse))]
        mse_data.append(mse)

        # Calculate statistical information about MSE
        mean_mse = np.nanmean(mse)
        std_dev_mse = np.nanstd(mse)
        min_mse = np.nanmin(mse)
        max_mse = np.nanmax(mse)

        # Display statistical information about MSE in the terminal
        print(f"Statistical Information about MSE ({method_name.upper()} - {patient_path} - pair bval):")
        print(f" - Mean: {mean_mse:.4f}")
        print(f" - Median: {np.nanmedian(mse):.4f}")
        print(f" - Standard Deviation: {std_dev_mse:.4f}")
        print(f" - Minimum: {min_mse}")
        print(f" - Maximum: {max_mse}")
        print()

    if method_name != "MF":
        mse, _ = load_nifti(method_path + f"{patient_path}_mse.nii.gz")

    else:
        mse, _ = load_nifti(method_path+f"{patient_path}_MSE.nii.gz")
    
    mse = mse[brain_mask_mask]
    mse = mse.flatten()
    mse = mse[np.logical_not(np.isnan(mse))]
    mse_data.append(mse)

    labels.append("all bvalues")

    # Plot boxplot of MSE for all b-values
    plt.figure(figsize=(8, 6))
    plt.boxplot(mse_data, showfliers=False, labels=labels, vert=True)
    plt.title(f"Boxplot MSE - {method_name.upper()} - {patient_path}")
    plt.xlabel("B-value")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    save_plot_path = f"{images_path}boxplot_mse_pair_{method_name}_{patient_path}.png"

    if not os.path.exists(os.path.dirname(save_plot_path)):
            os.makedirs(os.path.dirname(save_plot_path))

    if save_as_png:
        plt.savefig(save_plot_path, bbox_inches='tight')

    if show:
        plt.show()

    return


def calculate_and_plot_boxplot_mse_brain_white_matter(patient_path, method_name, prepoc_raw="preproc", show=False, save_as_png=False):
    """
    Calculate and plot the boxplot of Mean Squared Error (MSE) for multiple methods in white matter.

    Parameters:
    - patient_path (str): The name of the patient.
    - prepoc_raw (str): Specify whether to use preprocessed ("preproc") or raw ("raw") data.
    - show (bool): Whether to display the plot using plt.show(). Default is True.
    - save_as_png (bool): Whether to save the plot as a PNG file. Default is False.

    Returns:
    - None
    """
    # Define paths
    images_path = os.path.join(os.path.dirname(os.getcwd()), "images/mse")

    # Load brain and white matter masks
    brain_mask_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/masks/{patient_path}_brain_mask.nii.gz"
    brain_mask, _ = load_nifti(brain_mask_path)
    brain_mask_mask = brain_mask == 1 


    brain_mask_path_wm = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/masks/{patient_path}_wm_mask_AP.nii.gz"
    brain_mask_wm, _ = load_nifti(brain_mask_path_wm)
    white_matter_mask = brain_mask_wm == 1 

    if method_name == "mf":
        method_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/{patient_path}_MSE.nii.gz"
        mse, _ = load_nifti(method_path)
    else:
        method_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/{method_name}/"
        mse, _ = load_nifti(method_path + f"{patient_path}_mse.nii.gz")

    mse_wm = mse[white_matter_mask]
    mse_wm = mse_wm.flatten()
    mse_wm = mse_wm[np.logical_not(np.isnan(mse_wm))]

    mse_brain = mse[brain_mask_mask]
    mse_brain = mse_brain.flatten()
    mse_brain = mse_brain[np.logical_not(np.isnan(mse_brain))]

    # Calculate statistical information
    stats_brain = {
        "Mean": np.nanmean(mse_brain),
        "Median": np.nanmedian(mse_brain),
        "Standard Deviation": np.nanstd(mse_brain),
        "Minimum": np.nanmin(mse_brain),
        "Maximum": np.nanmax(mse_brain)
    }
    stats_wm = {
        "Mean": np.nanmean(mse_wm),
        "Median": np.nanmedian(mse_wm),
        "Standard Deviation": np.nanstd(mse_wm),
        "Minimum": np.nanmin(mse_wm),
        "Maximum": np.nanmax(mse_wm)
    }

    # Display statistical information
    print(f"Statistical Information about MSE ({method_name.upper()} - {patient_path}):")
    print(" - Full Brain:")
    for key, value in stats_brain.items():
        print(f"    - {key}: {value:.4f}")
    print(" - White Matter:")
    for key, value in stats_wm.items():
        print(f"    - {key}: {value:.4f}")
    print()

    # Plot boxplot
    plt.figure(figsize=(8, 6))
    plt.boxplot([mse_brain, mse_wm], labels=["Full Brain", "White Matter"],showfliers=False)
    plt.xlabel("Region")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title(f"Boxplot MSE - White Matter ({patient_path})")
    plt.grid(True, axis='y')

    # Save or show plot
    if save_as_png:
        plt.savefig(os.path.join(images_path, method_name, f"boxplot_mse_white_matter_brain_{patient_path}.png"), bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

    return


def calculate_and_plot_boxplot_mse_multi_b_values_1_fig_ROI(patient_path, method_name, atlas_name, atlas_values, roi_name, prepoc_raw="preproc", show=False, save_as_png=False):
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

    subject_folder = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}"

    images_path = folder_path + "/images/mse/" + method_name + "/"

    if method_name == "MF":
        method_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/"
    else:
        method_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/{method_name}/"

    brain_mask_path = f"{subject_folder}/reg/{patient_path}_Atlas_{atlas_name}_InSubjectDWISpaceFrom_AP.nii.gz"

    brain_mask, _ = load_nifti(brain_mask_path)

    brain_mask_mask = np.where(np.isin(brain_mask, atlas_values), 0, 1)

    if method_name == "DTI":
        b_values = [1000, 3000]
    else:
        b_values = [1000, 3000, 5000, 10000]

    mse_data = []

    for target_bvalue in b_values:

        if method_name == "DTI":
            mse, _ = load_nifti(method_path + f"{patient_path}_mse_b{target_bvalue}.nii.gz")
        
        elif method_name == "MF":
            print("MF")
            mse, _ = load_nifti(method_path+f"{patient_path}_b{target_bvalue}.0_MSE.nii.gz")
        else:
            mse, _ = load_nifti(method_path + f"{patient_path}_b{target_bvalue}_mse.nii.gz")

        mse = ma.masked_array(mse, mask=brain_mask_mask)
        mse = mse.compressed()
        mse = mse[np.logical_not(np.isnan(mse))]
        mse_data.append(mse)

        # Calculate statistical information about MSE
        mean_mse = np.nanmean(mse)
        std_dev_mse = np.nanstd(mse)
        min_mse = np.nanmin(mse)
        max_mse = np.nanmax(mse)


        # Display statistical information about MSE in the terminal
        print(f"Statistical Information about MSE ({method_name.upper()} - {patient_path} - B{target_bvalue}):")
        print(f" - Mean: {mean_mse:.4f}")
        print(f" - Median: {np.nanmedian(mse):.4f}")
        print(f" - Standard Deviation: {std_dev_mse:.4f}")
        print(f" - Minimum: {min_mse}")
        print(f" - Maximum: {max_mse}")
        print()
    if method_name != "MF":
        mse, _ = load_nifti(method_path + f"{patient_path}_mse.nii.gz")

    else:
        mse, _ = load_nifti(method_path+f"{patient_path}_MSE.nii.gz")
    
    mse = ma.masked_array(mse, mask=brain_mask_mask)
    mse = mse.compressed()
    mse = mse[np.logical_not(np.isnan(mse))]
    mse_data.append(mse)

    labels=[f"B{value}" for value in b_values]
    labels.append("all bvalues")

    # Plot boxplot of MSE for all b-values
    plt.figure(figsize=(8, 6))
    plt.boxplot(mse_data, showfliers=False, labels=labels, vert=True)
    plt.title(f"Boxplot MSE - {method_name.upper()} - {roi_name} - {patient_path}")
    plt.xlabel("B-value")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.grid(True, axis='y')
    plt.tight_layout()

    save_plot_path = f"{images_path}boxplot_mse_roi_{roi_name}_{method_name}_{patient_path}.png"

    if save_as_png:
        plt.savefig(save_plot_path, bbox_inches='tight')

    if show:
        plt.show()

    return





################################################################## REDUCED #############################################################
def plot_and_statistical_comparison_slices_reduced(patient_path, method_name, data_type,reduced_bval,reduced_nb, show=False, save_as_png=False):
    """
    This function, plot_and_statistical_comparison_slices, loads data for each b-value from MRI images and creates box-and-whisker plots to compare the data slices. It then performs statistical tests to determine the significance of differences between datasets.

    Arguments:
    - patient_path: Path to the patient's data directory.
    - method_name: Name of the MRI method (e.g., DTI or noddi).
    - data_type: Type of MRI data (e.g., FA or MD).
    - show: Boolean flag indicating whether to display the plots.
    - save_as_png: Boolean flag indicating whether to save the plots as PNG images.

    Description:
    - The function loads brain mask data and MRI data for each specified b-value from the provided patient's directory.
    - For each b-value, it extracts the data slices and filters them based on the brain mask.
    - It creates box-and-whisker plots to compare the filtered data slices.
    - Statistical significance tests (Kruskal-Wallis test) are performed to determine significant differences between datasets.
    - The plots are saved as PNG images if the 'save_as_png' flag is set to True.
    - The function also closes the plot after saving it to avoid memory accumulation.

    Note:
    - The function adjusts the position of the ylabel to prevent it from being cut off by the image.
    - Depending on the MRI method (DTI or noddi), different b-values are considered.
    - The last data slice for the overall MRI data (without specifying a b-value) is also included for comparison.

    Output:
    - Box-and-whisker plots comparing the filtered data slices.
    - Statistical significance bars indicating significant differences between datasets.
    - Sample size annotations below each box.
    - PNG images of the plots saved in the specified directory.
    """
    # Load data for each b-value
    data_slices = []
    filtered_data = []

    brain_mask_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/masks/{patient_path}_brain_mask.nii.gz"
    #brain_mask_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/masks/{patient_path}_wm_mask_AP.nii.gz"


    brain_mask, _ = load_nifti(brain_mask_path)

    brain_mask_mask = brain_mask == 1

    if method_name == "MF":
        data_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/"
    else:
        data_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/{method_name}/"
    
    
    data, _ = load_nifti(data_path + f"{patient_path}_{data_type}.nii.gz")
    filtered_data.append(data[brain_mask_mask].flatten())

    labels = ["all directions"]

    if type(reduced_nb) == list:
        for k in reduced_nb:
            data_reduced, _ = load_nifti(data_path + f"{patient_path}_reduced_b{reduced_bval}_{k}_{data_type}.nii.gz")
            filtered_data.append(data_reduced[brain_mask_mask].flatten())
            labels.append(f"{k}")
    else:
        data_reduced, _ = load_nifti(data_path + f"{patient_path}_reduced_b{reduced_bval}_{reduced_nb}_{data_type}.nii.gz")
        filtered_data.append(data_reduced[brain_mask_mask].flatten())
        labels.append("reduced")

    print("Data loaded successfully.")

    def box_and_whisker(data, title, ylabel, xticklabels):
        """
        Create a box-and-whisker plot with significance bars.
        """
        folder_path = os.path.abspath(os.path.dirname(os.getcwd()))

        fig, ax = plt.subplots()
        fig.subplots_adjust(left=0.2)
        bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False)
        # Graph title
        ax.set_title(title, fontsize=14)
        # Label y-axis
        ax.set_ylabel(ylabel)
        # Label x-axis ticks
        ax.set_xticks(range(1, len(xticklabels) + 1))  # Adjust ticks based on the number of labels
        ax.set_xticklabels(xticklabels)
        # Hide x-axis major ticks
        ax.tick_params(axis='x', which='major', length=0)
        # Show x-axis minor ticks
        xticks = [0.5] + [x + 0.5 for x in ax.get_xticks()]
        ax.set_xticks(xticks, minor=True)
        # Clean up the appearance
        ax.tick_params(axis='x', which='minor', length=3, width=1)

        # Change the colour of the boxes to Seaborn's 'pastel' palette
        colors = sns.color_palette('pastel')
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        # Colour of the median lines
        plt.setp(bp['medians'], color='k')

        # Check for statistical significance
        significant_combinations = []
        p_values = []

        # Check from the outside pairs of boxes inwards
        ls = list(range(1, len(data) + 1))
        combinations = [(ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))]
        for c in combinations:
            data1 = data[c[0] - 1]
            data2 = data[c[1] - 1]
            # Correct concatenation for the statistical test
            combined_data = np.concatenate((data1, data2))  # Note the double parentheses
            # Create a DataFrame suitable for pingouin's ANOVA
            groups = ['data1'] * len(data1) + ['data2'] * len(data2)  # Correctly repeat group labels
            df = pd.DataFrame({'score': combined_data, 'group': groups})
            # Perform the ANOVA
            aov = pg.anova(dv='score', between='group', data=df, detailed=True)
            # Append p-value and combination to lists if significant
            p_values.append(aov.at[0, 'p-unc'])
            if aov.at[0, 'p-unc'] < 0.05:  # Accessing p-value
                significant_combinations.append([c, aov.at[0, 'p-unc']])

        for i, (combination, p_val) in enumerate(zip(significant_combinations, p_values)):
            print(f"Significant difference between groups {combination[0]}: p-value = {p_val}")


        # Get info about y-axis
        bottom, top = ax.get_ylim()
        yrange = top - bottom

        # Significance bars
        for i, significant_combination in enumerate(significant_combinations):
            # Columns corresponding to the datasets of interest
            x1 = significant_combination[0][0]
            x2 = significant_combination[0][1]
            # What level is this bar among the bars above the plot?
            level = len(significant_combinations) - i
            # Plot the bar
            bar_height = (yrange * 0.08 * level) + top
            bar_tips = bar_height - (yrange * 0.02)
            plt.plot(
                [x1, x1, x2, x2],
                [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
            # Significance level
            p = significant_combination[1]
            if p < 0.001:
                sig_symbol = '***'
            elif p < 0.01:
                sig_symbol = '**'
            elif p < 0.05:
                sig_symbol = '*'
            text_height = bar_height + (yrange * 0.01)
            plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', c='k')

        # Adjust y-axis
        bottom, top = ax.get_ylim()
        yrange = top - bottom
        ax.set_ylim(bottom - 0.02 * yrange, top)

        # Annotate sample size below each box
        for i, dataset in enumerate(data):
            sample_size = len(dataset)
            ax.text(i + 1, bottom, fr'n = {sample_size}', ha='center', size='x-small')

        if show:
            plt.show()

        if True:
            save_path = folder_path + f"/images/{method}/{reduced_bval}/stats_comparison_reduced_b{reduced_bval}_{patient_path}_{method_name}_{data_type}.png"
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            plt.savefig(save_path)
            print(f"Plot saved as PNG: {save_path}")
        plt.close()
        return

    print("Creating box-and-whisker plot...")
    title = f"Boxplot Comparison of {data_type} Across {reduced_bval} b-values and Directions for {method_name} Method (Patient: {patient_path})"
    box_and_whisker(filtered_data, title, data_type, labels)
    print("Box-and-whisker plot created.")

    return


def plot_and_statistical_comparison_slices_reduced_ROI(patient_path,atlas_name, atlas_values, roi_name, method_name, data_type,reduced_bval,reduced_nb, show=False, save_as_png=False):
    """
    This function, plot_and_statistical_comparison_slices, loads data for each b-value from MRI images and creates box-and-whisker plots to compare the data slices. It then performs statistical tests to determine the significance of differences between datasets.

    Arguments:
    - patient_path: Path to the patient's data directory.
    - method_name: Name of the MRI method (e.g., DTI or noddi).
    - data_type: Type of MRI data (e.g., FA or MD).
    - show: Boolean flag indicating whether to display the plots.
    - save_as_png: Boolean flag indicating whether to save the plots as PNG images.

    Description:
    - The function loads brain mask data and MRI data for each specified b-value from the provided patient's directory.
    - For each b-value, it extracts the data slices and filters them based on the brain mask.
    - It creates box-and-whisker plots to compare the filtered data slices.
    - Statistical significance tests (Kruskal-Wallis test) are performed to determine significant differences between datasets.
    - The plots are saved as PNG images if the 'save_as_png' flag is set to True.
    - The function also closes the plot after saving it to avoid memory accumulation.

    Note:
    - The function adjusts the position of the ylabel to prevent it from being cut off by the image.
    - Depending on the MRI method (DTI or noddi), different b-values are considered.
    - The last data slice for the overall MRI data (without specifying a b-value) is also included for comparison.

    Output:
    - Box-and-whisker plots comparing the filtered data slices.
    - Statistical significance bars indicating significant differences between datasets.
    - Sample size annotations below each box.
    - PNG images of the plots saved in the specified directory.
    """
    # Load data for each b-value
    data_slices = []
    filtered_data = []

    subject_folder = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}"

    brain_mask_path = f"{subject_folder}/reg/{patient_path}_Atlas_{atlas_name}_InSubjectDWISpaceFrom_AP.nii.gz"

    brain_mask, _ = load_nifti(brain_mask_path)

    brain_mask_mask = np.where(np.isin(brain_mask, atlas_values), 0, 1)

    

    if method_name == "MF":
        data_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/"
    else:
        data_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/{method_name}/"
    
    
    data, _ = load_nifti(data_path + f"{patient_path}_{data_type}.nii.gz")
    data = ma.masked_array(data, mask=brain_mask_mask)
    filtered_data.append(data.compressed())

    labels = ["all directions"]

    if type(reduced_nb) == list:
        for k in reduced_nb:
            data_reduced, _ = load_nifti(data_path + f"{patient_path}_reduced_b{reduced_bval}_{k}_{data_type}.nii.gz")
            data = ma.masked_array(data_reduced, mask=brain_mask_mask)
            filtered_data.append(data.compressed())
            labels.append(f"{k}")
    else:
        data_reduced, _ = load_nifti(data_path + f"{patient_path}_reduced_b{reduced_bval}_{reduced_nb}_{data_type}.nii.gz")
        data = ma.masked_array(data_reduced, mask=brain_mask_mask)
        filtered_data.append(data.compressed())
        labels.append("reduced")

    print("Data loaded successfully.")

    def box_and_whisker(data, title, ylabel, xticklabels):
        """
        Create a box-and-whisker plot with significance bars.
        """
        folder_path = os.path.abspath(os.path.dirname(os.getcwd()))

        fig, ax = plt.subplots()
        fig.subplots_adjust(left=0.2)
        bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False)
        # Graph title
        ax.set_title(title, fontsize=14)
        # Label y-axis
        ax.set_ylabel(ylabel)
        # Label x-axis ticks
        ax.set_xticks(range(1, len(xticklabels) + 1))  # Adjust ticks based on the number of labels
        ax.set_xticklabels(xticklabels)
        # Hide x-axis major ticks
        ax.tick_params(axis='x', which='major', length=0)
        # Show x-axis minor ticks
        xticks = [0.5] + [x + 0.5 for x in ax.get_xticks()]
        ax.set_xticks(xticks, minor=True)
        # Clean up the appearance
        ax.tick_params(axis='x', which='minor', length=3, width=1)

        # Change the colour of the boxes to Seaborn's 'pastel' palette
        colors = sns.color_palette('pastel')
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        # Colour of the median lines
        plt.setp(bp['medians'], color='k')

        # Check for statistical significance
        significant_combinations = []
        p_values = []

        # Check from the outside pairs of boxes inwards
        ls = list(range(1, len(data) + 1))
        combinations = [(ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))]
        for c in combinations:
            data1 = data[c[0] - 1]
            data2 = data[c[1] - 1]
            # Correct concatenation for the statistical test
            combined_data = np.concatenate((data1, data2))  # Note the double parentheses
            # Create a DataFrame suitable for pingouin's ANOVA
            groups = ['data1'] * len(data1) + ['data2'] * len(data2)  # Correctly repeat group labels
            df = pd.DataFrame({'score': combined_data, 'group': groups})
            # Perform the ANOVA
            aov = pg.anova(dv='score', between='group', data=df, detailed=True)
            # Append p-value and combination to lists if significant
            p_values.append(aov.at[0, 'p-unc'])
            if aov.at[0, 'p-unc'] < 0.05:  # Accessing p-value
                significant_combinations.append([c, aov.at[0, 'p-unc']])

        for i, (combination, p_val) in enumerate(zip(significant_combinations, p_values)):
            print(f"Significant difference between groups {combination[0]}: p-value = {p_val}")


        # Get info about y-axis
        bottom, top = ax.get_ylim()
        yrange = top - bottom

        # Significance bars
        for i, significant_combination in enumerate(significant_combinations):
            # Columns corresponding to the datasets of interest
            x1 = significant_combination[0][0]
            x2 = significant_combination[0][1]
            # What level is this bar among the bars above the plot?
            level = len(significant_combinations) - i
            # Plot the bar
            bar_height = (yrange * 0.08 * level) + top
            bar_tips = bar_height - (yrange * 0.02)
            plt.plot(
                [x1, x1, x2, x2],
                [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
            # Significance level
            p = significant_combination[1]
            if p < 0.001:
                sig_symbol = '***'
            elif p < 0.01:
                sig_symbol = '**'
            elif p < 0.05:
                sig_symbol = '*'
            text_height = bar_height + (yrange * 0.01)
            plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', c='k')

        # Adjust y-axis
        bottom, top = ax.get_ylim()
        yrange = top - bottom
        ax.set_ylim(bottom - 0.02 * yrange, top)

        # Annotate sample size below each box
        for i, dataset in enumerate(data):
            sample_size = len(dataset)
            ax.text(i + 1, bottom, fr'n = {sample_size}', ha='center', size='x-small')

        if show:
            plt.show()

        if True:
            save_path = folder_path + f"/images/{method}/{reduced_bval}/{roi_name}/stats_comparison_reduced_{roi_name}_b{reduced_bval}_{patient_path}_{method_name}_{data_type}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            plt.savefig(save_path)
            print(f"Plot saved as PNG: {save_path}")
        plt.close()
        return

    print("Creating box-and-whisker plot...")
    title = f"Comparison of {data_type} Data for {patient_path} with direction for {reduced_bval} (Method: {method_name})"
    box_and_whisker(filtered_data, title, data_type, labels)
    print("Box-and-whisker plot created.")

    return


######################################################################### Pair ######################################################


def compare_MRI_bvalues_pair_statistically(patient_path, method_name, data_type, preprocessing_type, show=False, save_as_png=False):
    """
    This function, plot_and_statistical_comparison_slices, loads data for each b-value from MRI images and creates box-and-whisker plots to compare the data slices. It then performs statistical tests to determine the significance of differences between datasets.

    Arguments:
    - patient_path: Path to the patient's data directory.
    - method_name: Name of the MRI method (e.g., DTI or noddi).
    - data_type: Type of MRI data (e.g., FA or MD).
    - preprocessing_type: Type of preprocessing applied to the data.
    - show: Boolean flag indicating whether to display the plots.
    - save_as_png: Boolean flag indicating whether to save the plots as PNG images.

    Description:
    - The function loads brain mask data and MRI data for each specified b-value from the provided patient's directory.
    - For each b-value, it extracts the data slices and filters them based on the brain mask.
    - It creates box-and-whisker plots to compare the filtered data slices.
    - Statistical significance tests (Kruskal-Wallis test) are performed to determine significant differences between datasets.
    - The plots are saved as PNG images if the 'save_as_png' flag is set to True.
    - The function also closes the plot after saving it to avoid memory accumulation.

    Note:
    - The function adjusts the position of the ylabel to prevent it from being cut off by the image.
    - Depending on the MRI method (DTI or noddi), different b-values are considered.
    - The last data slice for the overall MRI data (without specifying a b-value) is also included for comparison.

    Output:
    - Box-and-whisker plots comparing the filtered data slices.
    - Statistical significance bars indicating significant differences between datasets.
    - Sample size annotations below each box.
    - PNG images of the plots saved in the specified directory.
    """
    # Load data for each b-value
    filtered_data = []

    brain_mask_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/masks/{patient_path}_brain_mask.nii.gz"
    #brain_mask_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/masks/{patient_path}_wm_mask_AP.nii.gz"

    brain_mask, _ = load_nifti(brain_mask_path)

    brain_mask_mask = brain_mask == 1

    if method_name == "DTI":
        b_values = [1000, 3000]
    else:
        b_values = [1000, 3000, 5000, 10000]

    bval_combinations = [(b1, b2) for i, b1 in enumerate(b_values) for b2 in b_values[i+1:] if b1 != b2]

    data_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/{method}/"
    
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

        filtered_data.append(data[brain_mask_mask].flatten())
    
    if method_name =="MF":
        data, _ = load_nifti(f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/{patient_path}_{data_type}.nii.gz")
    else:   
        data, _ = load_nifti(data_path + f"{patient_path}_{data_type}.nii.gz")
    filtered_data.append(data[brain_mask_mask].flatten())

    print("Data loaded successfully.")

    labels.append("all bvals")

    def box_and_whisker(data, title, ylabel, xticklabels, method_name, data_type, preprocessing_type="preproc"):
        """
        Create a box-and-whisker plot with significance bars.
        """
        folder_path = os.path.abspath(os.path.dirname(os.getcwd()))

        fig, ax = plt.subplots(figsize=(20,20))
        fig.subplots_adjust(left=0.2)
        bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False)
        # Graph title
        ax.set_title(title, fontsize=14)
        # Label y-axis
        ax.set_ylabel(ylabel)
        # Label x-axis ticks
        ax.set_xticks(range(1, len(xticklabels) + 1))  # Adjust ticks based on the number of labels
        ax.set_xticklabels(xticklabels)
        # Hide x-axis major ticks
        ax.tick_params(axis='x', which='major', length=0)
        # Show x-axis minor ticks
        xticks = [0.5] + [x + 0.5 for x in ax.get_xticks()]
        ax.set_xticks(xticks, minor=True)
        # Clean up the appearance
        ax.tick_params(axis='x', which='minor', length=3, width=1)

        # Change the colour of the boxes to Seaborn's 'pastel' palette
        colors = sns.color_palette('pastel')
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        # Colour of the median lines
        plt.setp(bp['medians'], color='k')

        # Check for statistical significance
        significant_combinations = []
        p_values = []

        # Check from the outside pairs of boxes inwards
        ls = list(range(1, len(data) + 1))
        combinations = [(ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))]
        for c in combinations:
            data1 = data[c[0] - 1]
            data2 = data[c[1] - 1]
            # Correct concatenation for the statistical test
            combined_data = np.concatenate((data1, data2))  # Note the double parentheses
            # Create a DataFrame suitable for pingouin's ANOVA
            groups = ['data1'] * len(data1) + ['data2'] * len(data2)  # Correctly repeat group labels
            df = pd.DataFrame({'score': combined_data, 'group': groups})
            # Perform the ANOVA
            aov = pg.anova(dv='score', between='group', data=df, detailed=True)
            # Append p-value and combination to lists if significant
            p_values.append(aov.at[0, 'p-unc'])
            if aov.at[0, 'p-unc'] < 0.05:  # Accessing p-value
                significant_combinations.append([c, aov.at[0, 'p-unc']])
        
        for i, (combination, p_val) in enumerate(zip(significant_combinations, p_values)):
            print(f"Significant difference between groups {combination[0]}: p-value = {p_val}")


        # Get info about y-axis
        bottom, top = ax.get_ylim()
        yrange = top - bottom

        # Significance bars
        for i, significant_combination in enumerate(significant_combinations):
            # Columns corresponding to the datasets of interest
            x1 = significant_combination[0][0]
            x2 = significant_combination[0][1]
            # What level is this bar among the bars above the plot?
            level = len(significant_combinations) - i
            # Plot the bar
            bar_height = (yrange * 0.08 * level) + top
            bar_tips = bar_height - (yrange * 0.02)
            plt.plot(
                [x1, x1, x2, x2],
                [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
            # Significance level
            p = significant_combination[1]
            if p < 0.001:
                sig_symbol = '***'
            elif p < 0.01:
                sig_symbol = '**'
            elif p < 0.05:
                sig_symbol = '*'
            text_height = bar_height + (yrange * 0.01)
            plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', c='k')

        # Adjust y-axis
        bottom, top = ax.get_ylim()
        yrange = top - bottom
        ax.set_ylim(bottom - 0.02 * yrange, top)

        # Annotate sample size below each box
        for i, dataset in enumerate(data):
            sample_size = len(dataset)
            ax.text(i + 1, bottom, fr'n = {sample_size}', ha='center', size='x-small')


        save_path = folder_path + f"/images/{method}/stats_comparison_pair_{patient_path}_{method_name}_{data_type}_{preprocessing_type}.png"
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path)
        print(f"Plot saved as PNG: {save_path}")
        plt.close()
        return

    print("Creating box-and-whisker plot...")
    if method_name =="DTI":
        title = f"Comparison of {data_type} Data for {patient_path} (Method: {method_name})"
        box_and_whisker(filtered_data, title, data_type, labels,method_name,data_type)
    else :
        title = f"Comparison of {data_type} Data for {patient_path} (Method: {method_name})"
        box_and_whisker(filtered_data, title, data_type, labels ,method_name,data_type)
    print("Box-and-whisker plot created.")

    return


def compare_MRI_bvalues_pair_ROI_statistically(patient_path,atlas_name, atlas_values, roi_name, method_name, data_type, show=False, save_as_png=False):
    """
    This function, plot_and_statistical_comparison_slices, loads data for each b-value from MRI images and creates box-and-whisker plots to compare the data slices. It then performs statistical tests to determine the significance of differences between datasets.

    Arguments:
    - patient_path: Path to the patient's data directory.
    - method_name: Name of the MRI method (e.g., DTI or noddi).
    - data_type: Type of MRI data (e.g., FA or MD).
    - preprocessing_type: Type of preprocessing applied to the data.
    - show: Boolean flag indicating whether to display the plots.
    - save_as_png: Boolean flag indicating whether to save the plots as PNG images.

    Description:
    - The function loads brain mask data and MRI data for each specified b-value from the provided patient's directory.
    - For each b-value, it extracts the data slices and filters them based on the brain mask.
    - It creates box-and-whisker plots to compare the filtered data slices.
    - Statistical significance tests (Kruskal-Wallis test) are performed to determine significant differences between datasets.
    - The plots are saved as PNG images if the 'save_as_png' flag is set to True.
    - The function also closes the plot after saving it to avoid memory accumulation.

    Note:
    - The function adjusts the position of the ylabel to prevent it from being cut off by the image.
    - Depending on the MRI method (DTI or noddi), different b-values are considered.
    - The last data slice for the overall MRI data (without specifying a b-value) is also included for comparison.

    Output:
    - Box-and-whisker plots comparing the filtered data slices.
    - Statistical significance bars indicating significant differences between datasets.
    - Sample size annotations below each box.
    - PNG images of the plots saved in the specified directory.
    """
    subject_folder = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}"

    # Load data for each b-value
    filtered_data = []

    brain_mask_path = f"{subject_folder}/reg/{patient_path}_Atlas_{atlas_name}_InSubjectDWISpaceFrom_AP.nii.gz"

    brain_mask, _ = load_nifti(brain_mask_path)

    brain_mask_mask = np.where(np.isin(brain_mask, atlas_values), 0, 1)

    if method_name == "DTI":
        b_values = [1000, 3000]
    else:
        b_values = [1000, 3000, 5000, 10000]

    bval_combinations = [(b1, b2) for i, b1 in enumerate(b_values) for b2 in b_values[i+1:] if b1 != b2]

    data_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/{method}/"
    
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
        
        data = ma.masked_array(data, mask=brain_mask_mask)
        
        print(f"Loading data from: {data_path}")

        filtered_data.append(data.compressed())
    
    if method_name =="MF":
        data, _ = load_nifti(f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/{patient_path}_{data_type}.nii.gz")
    else:   
        data, _ = load_nifti(data_path + f"{patient_path}_{data_type}.nii.gz")
    data = ma.masked_array(data, mask=brain_mask_mask)
    filtered_data.append(data.compressed())

    print("Data loaded successfully.")

    labels.append("all bvals")

    def box_and_whisker(data, title, ylabel, xticklabels, method_name, data_type, preprocessing_type="preproc"):
        """
        Create a box-and-whisker plot with significance bars.
        """
        folder_path = os.path.abspath(os.path.dirname(os.getcwd()))

        fig, ax = plt.subplots(figsize=(20,20))
        fig.subplots_adjust(left=0.2)
        bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False)
        # Graph title
        ax.set_title(title, fontsize=14)
        # Label y-axis
        ax.set_ylabel(ylabel)
        # Label x-axis ticks
        ax.set_xticks(range(1, len(xticklabels) + 1))  # Adjust ticks based on the number of labels
        ax.set_xticklabels(xticklabels)
        # Hide x-axis major ticks
        ax.tick_params(axis='x', which='major', length=0)
        # Show x-axis minor ticks
        xticks = [0.5] + [x + 0.5 for x in ax.get_xticks()]
        ax.set_xticks(xticks, minor=True)
        # Clean up the appearance
        ax.tick_params(axis='x', which='minor', length=3, width=1)

        # Change the colour of the boxes to Seaborn's 'pastel' palette
        colors = sns.color_palette('pastel')
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        # Colour of the median lines
        plt.setp(bp['medians'], color='k')

        # Check for statistical significance
        significant_combinations = []
        p_values = []

        # Check from the outside pairs of boxes inwards
        ls = list(range(1, len(data) + 1))
        combinations = [(ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))]
        for c in combinations:
            data1 = data[c[0] - 1]
            data2 = data[c[1] - 1]
            # Correct concatenation for the statistical test
            combined_data = np.concatenate((data1, data2))  # Note the double parentheses
            # Create a DataFrame suitable for pingouin's ANOVA
            groups = ['data1'] * len(data1) + ['data2'] * len(data2)  # Correctly repeat group labels
            df = pd.DataFrame({'score': combined_data, 'group': groups})
            # Perform the ANOVA
            aov = pg.anova(dv='score', between='group', data=df, detailed=True)

            if 'p-unc' in aov.columns:
                p_values.append(aov.at[0, 'p-unc'])
            if aov.at[0, 'p-unc'] < 0.05:  # Accessing p-value
                significant_combinations.append([c, aov.at[0, 'p-unc']])
            else:
                print(f"Warning: 'p-unc' not found in ANOVA result columns: {aov.columns}")

        for i, (combination, p_val) in enumerate(zip(significant_combinations, p_values)):
            print(f"Significant difference between groups {combination[0]}: p-value = {p_val}")


        # Get info about y-axis
        bottom, top = ax.get_ylim()
        yrange = top - bottom

        # Significance bars
        for i, significant_combination in enumerate(significant_combinations):
            # Columns corresponding to the datasets of interest
            x1 = significant_combination[0][0]
            x2 = significant_combination[0][1]
            # What level is this bar among the bars above the plot?
            level = len(significant_combinations) - i
            # Plot the bar
            bar_height = (yrange * 0.08 * level) + top
            bar_tips = bar_height - (yrange * 0.02)
            plt.plot(
                [x1, x1, x2, x2],
                [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
            # Significance level
            p = significant_combination[1]
            if p < 0.001:
                sig_symbol = '***'
            elif p < 0.01:
                sig_symbol = '**'
            elif p < 0.05:
                sig_symbol = '*'
            text_height = bar_height + (yrange * 0.01)
            plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', c='k')

        # Adjust y-axis
        bottom, top = ax.get_ylim()
        yrange = top - bottom
        ax.set_ylim(bottom - 0.02 * yrange, top)

        # Annotate sample size below each box
        for i, dataset in enumerate(data):
            sample_size = len(dataset)
            ax.text(i + 1, bottom, fr'n = {sample_size}', ha='center', size='x-small')


        save_path = folder_path + f"/images/{method}/{roi_name}/stats_comparison_{roi_name}_pair_{patient_path}_{method_name}_{data_type}_{preprocessing_type}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path)
        print(f"Plot saved as PNG: {save_path}")
        plt.close()
        return

    print("Creating box-and-whisker plot...")
    if method_name =="DTI":
        title = f"Comparison of {data_type} Data for {patient_path} (Method: {method_name})"
        box_and_whisker(filtered_data, title, data_type, labels,method_name,data_type)
    else :
        title = f"Comparison of {data_type} Data for {patient_path} (Method: {method_name})"
        box_and_whisker(filtered_data, title, data_type, labels ,method_name,data_type)
    print("Box-and-whisker plot created.")

    return


######################################################################### Stats ######################################################
def box_and_whisker(data, title, ylabel, xticklabels, method_name, data_type, preprocessing_type="preproc"):
        """
        Create a box-and-whisker plot with significance bars.
        """
        folder_path = os.path.abspath(os.path.dirname(os.getcwd()))

        fig, ax = plt.subplots()
        fig.subplots_adjust(left=0.2)
        bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False)
        # Graph title
        ax.set_title(title, fontsize=14)
        # Label y-axis
        ax.set_ylabel(ylabel)
        # Label x-axis ticks
        ax.set_xticks(range(1, len(xticklabels) + 1))  # Adjust ticks based on the number of labels
        ax.set_xticklabels(xticklabels)
        # Hide x-axis major ticks
        ax.tick_params(axis='x', which='major', length=0)
        # Show x-axis minor ticks
        xticks = [0.5] + [x + 0.5 for x in ax.get_xticks()]
        ax.set_xticks(xticks, minor=True)
        # Clean up the appearance
        ax.tick_params(axis='x', which='minor', length=3, width=1)

        # Change the colour of the boxes to Seaborn's 'pastel' palette
        colors = sns.color_palette('pastel')
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        # Colour of the median lines
        plt.setp(bp['medians'], color='k')

        # Check for statistical significance
        significant_combinations = []
        p_values = []

        # Check from the outside pairs of boxes inwards
        ls = list(range(1, len(data) + 1))
        combinations = [(ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))]
        for c in combinations:
            data1 = data[c[0] - 1]
            data2 = data[c[1] - 1]
            # Correct concatenation for the statistical test
            combined_data = np.concatenate((data1, data2))  # Note the double parentheses
            # Create a DataFrame suitable for pingouin's ANOVA
            groups = ['data1'] * len(data1) + ['data2'] * len(data2)  # Correctly repeat group labels
            df = pd.DataFrame({'score': combined_data, 'group': groups})
            # Perform the ANOVA
            aov = pg.anova(dv='score', between='group', data=df, detailed=True)
            # Append p-value and combination to lists if significant
            p_values.append(aov.at[0, 'p-unc'])
            if aov.at[0, 'p-unc'] < 0.05:  # Accessing p-value
                significant_combinations.append([c, aov.at[0, 'p-unc']])
        
        for i, (combination, p_val) in enumerate(zip(significant_combinations, p_values)):
            print(f"Significant difference between groups {combination[0]}: p-value = {p_val}")


        # Get info about y-axis
        bottom, top = ax.get_ylim()
        yrange = top - bottom

        # Significance bars
        for i, significant_combination in enumerate(significant_combinations):
            # Columns corresponding to the datasets of interest
            x1 = significant_combination[0][0]
            x2 = significant_combination[0][1]
            # What level is this bar among the bars above the plot?
            level = len(significant_combinations) - i
            # Plot the bar
            bar_height = (yrange * 0.08 * level) + top
            bar_tips = bar_height - (yrange * 0.02)
            plt.plot(
                [x1, x1, x2, x2],
                [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
            # Significance level
            p = significant_combination[1]
            if p < 0.001:
                sig_symbol = '***'
            elif p < 0.01:
                sig_symbol = '**'
            elif p < 0.05:
                sig_symbol = '*'
            text_height = bar_height + (yrange * 0.01)
            plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', c='k')

        # Adjust y-axis
        bottom, top = ax.get_ylim()
        yrange = top - bottom
        ax.set_ylim(bottom - 0.02 * yrange, top)

        # Annotate sample size below each box
        for i, dataset in enumerate(data):
            sample_size = len(dataset)
            ax.text(i + 1, bottom, fr'n = {sample_size}', ha='center', size='x-small')


        save_path = folder_path + f"/images/{method}/stats_comparison_{patient_path}_{method_name}_{data_type}_{preprocessing_type}.png"
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path)
        print(f"Plot saved as PNG: {save_path}")
        plt.close()
        return

def compare_MRI_bvalues_statistically(patient_path, method_name, data_type, preprocessing_type, show=False, save_as_png=False):
    """
    This function, plot_and_statistical_comparison_slices, loads data for each b-value from MRI images and creates box-and-whisker plots to compare the data slices. It then performs statistical tests to determine the significance of differences between datasets.

    Arguments:
    - patient_path: Path to the patient's data directory.
    - method_name: Name of the MRI method (e.g., DTI or noddi).
    - data_type: Type of MRI data (e.g., FA or MD).
    - preprocessing_type: Type of preprocessing applied to the data.
    - show: Boolean flag indicating whether to display the plots.
    - save_as_png: Boolean flag indicating whether to save the plots as PNG images.

    Description:
    - The function loads brain mask data and MRI data for each specified b-value from the provided patient's directory.
    - For each b-value, it extracts the data slices and filters them based on the brain mask.
    - It creates box-and-whisker plots to compare the filtered data slices.
    - Statistical significance tests (Kruskal-Wallis test) are performed to determine significant differences between datasets.
    - The plots are saved as PNG images if the 'save_as_png' flag is set to True.
    - The function also closes the plot after saving it to avoid memory accumulation.

    Note:
    - The function adjusts the position of the ylabel to prevent it from being cut off by the image.
    - Depending on the MRI method (DTI or noddi), different b-values are considered.
    - The last data slice for the overall MRI data (without specifying a b-value) is also included for comparison.

    Output:
    - Box-and-whisker plots comparing the filtered data slices.
    - Statistical significance bars indicating significant differences between datasets.
    - Sample size annotations below each box.
    - PNG images of the plots saved in the specified directory.
    """
    # Load data for each b-value
    filtered_data = []

    brain_mask_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/masks/{patient_path}_brain_mask.nii.gz"
    #brain_mask_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/masks/{patient_path}_wm_mask_AP.nii.gz"

    brain_mask, _ = load_nifti(brain_mask_path)

    brain_mask_mask = brain_mask == 1

    if method_name == "DTI":
        b_values = [1000, 3000]

    elif method_name == "MF":
        b_values = [1000, 3000, 5000]
    else:
        b_values = [1000, 3000, 5000, 10000]

    data_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/{method}/"
    
    for b_value in b_values:
        if method_name == "noddi":
            data, _ = load_nifti(data_path + f"{patient_path}_b{b_value}_{data_type}.nii.gz")
        elif method_name == "MF":
            data, _ = load_nifti(f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/{patient_path}_b{b_value}_{data_type}.nii.gz")
        else:
            data, _ = load_nifti(data_path + f"{patient_path}_{data_type}_b{b_value}.nii.gz")
        
        print(f"Loading data from: {data_path}")

        filtered_data.append(data[brain_mask_mask].flatten())
    
    if method_name =="MF":
        data, _ = load_nifti(f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/{patient_path}_{data_type}.nii.gz")
    else:   
        data, _ = load_nifti(data_path + f"{patient_path}_{data_type}.nii.gz")
    filtered_data.append(data[brain_mask_mask].flatten())

    print("Data loaded successfully.")

    print("Creating box-and-whisker plot...")
    if method_name =="DTI":
        title = f"Comparison of {data_type} Data for {patient_path} (Method: {method_name})"
        box_and_whisker(filtered_data, title, data_type, ["B1000","B3000","All bvlaues"],method_name,data_type)
    else :
        title = f"Comparison of {data_type} Data for {patient_path} (Method: {method_name})"
        box_and_whisker(filtered_data, title, data_type, ["B1000","B3000","B5000","B10000","All bvlaues"],method_name,data_type)
    print("Box-and-whisker plot created.")

    return



def compare_MRI_bvalues_ROI_statistically(patient_path,atlas_name, atlas_values, roi_name, method_name, data_type, show=False, save_as_png=False):
    """
    This function, plot_and_statistical_comparison_slices, loads data for each b-value from MRI images and creates box-and-whisker plots to compare the data slices. It then performs statistical tests to determine the significance of differences between datasets.

    Arguments:
    - patient_path: Path to the patient's data directory.
    - method_name: Name of the MRI method (e.g., DTI or noddi).
    - data_type: Type of MRI data (e.g., FA or MD).
    - preprocessing_type: Type of preprocessing applied to the data.
    - show: Boolean flag indicating whether to display the plots.
    - save_as_png: Boolean flag indicating whether to save the plots as PNG images.

    Description:
    - The function loads brain mask data and MRI data for each specified b-value from the provided patient's directory.
    - For each b-value, it extracts the data slices and filters them based on the brain mask.
    - It creates box-and-whisker plots to compare the filtered data slices.
    - Statistical significance tests (Kruskal-Wallis test) are performed to determine significant differences between datasets.
    - The plots are saved as PNG images if the 'save_as_png' flag is set to True.
    - The function also closes the plot after saving it to avoid memory accumulation.

    Note:
    - The function adjusts the position of the ylabel to prevent it from being cut off by the image.
    - Depending on the MRI method (DTI or noddi), different b-values are considered.
    - The last data slice for the overall MRI data (without specifying a b-value) is also included for comparison.

    Output:
    - Box-and-whisker plots comparing the filtered data slices.
    - Statistical significance bars indicating significant differences between datasets.
    - Sample size annotations below each box.
    - PNG images of the plots saved in the specified directory.
    """
    subject_folder = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}"

    # Load data for each b-value
    filtered_data = []

    brain_mask_path = f"{subject_folder}/reg/{patient_path}_Atlas_{atlas_name}_InSubjectDWISpaceFrom_AP.nii.gz"

    brain_mask, _ = load_nifti(brain_mask_path)

    brain_mask_mask = np.where(np.isin(brain_mask, atlas_values), 0, 1)

    if method_name == "DTI":
        b_values = [1000, 3000]

    elif  method_name == "MF":
        b_values = [1000, 3000, 5000]
    else:
        b_values = [1000, 3000, 5000, 10000]

    data_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/{method}/"
    
    labels = []
    for bval in b_values:
        labels.append(f'{bval}')
        if method_name == "noddi":
            data, _ = load_nifti(data_path + f"{patient_path}_b{bval}_{data_type}.nii.gz")
        elif method_name == "MF":
            data, _ = load_nifti(f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/{patient_path}_b{bval}_{data_type}.nii.gz")
        else:
            data, _ = load_nifti(data_path + f"{patient_path}_{data_type}_b{bval}.nii.gz")
        
        data = ma.masked_array(data, mask=brain_mask_mask)
        
        print(f"Loading data from: {data_path}")

        filtered_data.append(data.compressed())
    
    if method_name =="MF":
        data, _ = load_nifti(f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/{patient_path}_{data_type}.nii.gz")
    else:   
        data, _ = load_nifti(data_path + f"{patient_path}_{data_type}.nii.gz")
    data = ma.masked_array(data, mask=brain_mask_mask)
    filtered_data.append(data.compressed())

    print("Data loaded successfully.")

    labels.append("all bvals")

    def box_and_whisker(data, title, ylabel, xticklabels, method_name, data_type, preprocessing_type="preproc"):
        """
        Create a box-and-whisker plot with significance bars.
        """
        folder_path = os.path.abspath(os.path.dirname(os.getcwd()))

        fig, ax = plt.subplots(figsize=(20,20))
        fig.subplots_adjust(left=0.2)
        bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False)
        # Graph title
        ax.set_title(title, fontsize=14)
        # Label y-axis
        ax.set_ylabel(ylabel)
        # Label x-axis ticks
        ax.set_xticks(range(1, len(xticklabels) + 1))  # Adjust ticks based on the number of labels
        ax.set_xticklabels(xticklabels)
        # Hide x-axis major ticks
        ax.tick_params(axis='x', which='major', length=0)
        # Show x-axis minor ticks
        xticks = [0.5] + [x + 0.5 for x in ax.get_xticks()]
        ax.set_xticks(xticks, minor=True)
        # Clean up the appearance
        ax.tick_params(axis='x', which='minor', length=3, width=1)

        # Change the colour of the boxes to Seaborn's 'pastel' palette
        colors = sns.color_palette('pastel')
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        # Colour of the median lines
        plt.setp(bp['medians'], color='k')

        # Check for statistical significance
        significant_combinations = []
        p_values = []

        # Check from the outside pairs of boxes inwards
        ls = list(range(1, len(data) + 1))
        combinations = [(ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))]
        for c in combinations:
            data1 = data[c[0] - 1]
            data2 = data[c[1] - 1]
            # Correct concatenation for the statistical test
            combined_data = np.concatenate((data1, data2))  # Note the double parentheses
            # Create a DataFrame suitable for pingouin's ANOVA
            groups = ['data1'] * len(data1) + ['data2'] * len(data2)  # Correctly repeat group labels
            df = pd.DataFrame({'score': combined_data, 'group': groups})
            # Perform the ANOVA
            aov = pg.anova(dv='score', between='group', data=df, detailed=True)
            # Append p-value and combination to lists if significant
            p_values.append(aov.at[0, 'p-unc'])
            if aov.at[0, 'p-unc'] < 0.05:  # Accessing p-value
                significant_combinations.append([c, aov.at[0, 'p-unc']])
        
        for i, (combination, p_val) in enumerate(zip(significant_combinations, p_values)):
            print(f"Significant difference between groups {combination[0]}: p-value = {p_val}")


        # Get info about y-axis
        bottom, top = ax.get_ylim()
        yrange = top - bottom

        # Significance bars
        for i, significant_combination in enumerate(significant_combinations):
            # Columns corresponding to the datasets of interest
            x1 = significant_combination[0][0]
            x2 = significant_combination[0][1]
            # What level is this bar among the bars above the plot?
            level = len(significant_combinations) - i
            # Plot the bar
            bar_height = (yrange * 0.08 * level) + top
            bar_tips = bar_height - (yrange * 0.02)
            plt.plot(
                [x1, x1, x2, x2],
                [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
            # Significance level
            p = significant_combination[1]
            if p < 0.001:
                sig_symbol = '***'
            elif p < 0.01:
                sig_symbol = '**'
            elif p < 0.05:
                sig_symbol = '*'
            text_height = bar_height + (yrange * 0.01)
            plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', c='k')

        # Adjust y-axis
        bottom, top = ax.get_ylim()
        yrange = top - bottom
        ax.set_ylim(bottom - 0.02 * yrange, top)

        # Annotate sample size below each box
        for i, dataset in enumerate(data):
            sample_size = len(dataset)
            ax.text(i + 1, bottom, fr'n = {sample_size}', ha='center', size='x-small')


        save_path = folder_path + f"/images/{method}/{roi_name}/stats_comparison_{roi_name}_{patient_path}_{method_name}_{data_type}_{preprocessing_type}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path)
        print(f"Plot saved as PNG: {save_path}")
        plt.close()
        return

    print("Creating box-and-whisker plot...")
    if method_name =="DTI":
        title = f"Comparison of {data_type} Data for {patient_path} (Method: {method_name})"
        box_and_whisker(filtered_data, title, data_type, labels,method_name,data_type)
    else :
        title = f"Comparison of {data_type} Data for {patient_path} (Method: {method_name})"
        box_and_whisker(filtered_data, title, data_type, labels ,method_name,data_type)
    print("Box-and-whisker plot created.")

    return


def compare_nb_bundles(method_name, data_type,nb_direction):


    patient_path ="sub-1007"

    subject_folder = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}"

    # Load data for each b-value
    filtered_data = []

    if nb_direction == 1:
        brain_mask_path = f"../Registration/mask_{nb_direction}_direction.nii.gz"
    else:
        brain_mask_path = f"../Registration/mask_{nb_direction}_directions.nii.gz"

    brain_mask, _ = load_nifti(brain_mask_path)

    brain_mask_mask = np.where(np.isin(brain_mask, [1]), 0, 1)

    if method_name == "DTI":
        b_values = [1000, 3000]

    elif  method_name == "MF":
        b_values = [1000, 3000, 5000]
    else:
        b_values = [1000, 3000, 5000, 10000]

    data_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/{method_name}/"
    
    labels = []
    for bval in b_values:
        labels.append(f'{bval}')
        if method_name == "noddi":
            data, _ = load_nifti(data_path + f"{patient_path}_b{bval}_{data_type}.nii.gz")
        elif method_name == "MF":
            data, _ = load_nifti(f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/{patient_path}_b{bval}_{data_type}.nii.gz")
        else:
            data, _ = load_nifti(data_path + f"{patient_path}_{data_type}_b{bval}.nii.gz")
        
        data = ma.masked_array(data, mask=brain_mask_mask)
        
        print(f"Loading data from: {data_path}")

        filtered_data.append(data.compressed())
    
    if method_name =="MF":
        data, _ = load_nifti(f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{patient_path}/dMRI/microstructure/mf/{patient_path}_{data_type}.nii.gz")
    else:   
        data, _ = load_nifti(data_path + f"{patient_path}_{data_type}.nii.gz")
    data = ma.masked_array(data, mask=brain_mask_mask)
    filtered_data.append(data.compressed())

    print("Data loaded successfully.")

    labels.append("all bvals")

    def box_and_whisker(data, title, ylabel, xticklabels, method_name, data_type, preprocessing_type="preproc"):
        """
        Create a box-and-whisker plot with significance bars.
        """
        folder_path = os.path.abspath(os.path.dirname(os.getcwd()))

        fig, ax = plt.subplots(figsize=(20,20))
        fig.subplots_adjust(left=0.2)
        bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False)
        # Graph title
        ax.set_title(title, fontsize=14)
        # Label y-axis
        ax.set_ylabel(ylabel)
        # Label x-axis ticks
        ax.set_xticks(range(1, len(xticklabels) + 1))  # Adjust ticks based on the number of labels
        ax.set_xticklabels(xticklabels)
        # Hide x-axis major ticks
        ax.tick_params(axis='x', which='major', length=0)
        # Show x-axis minor ticks
        xticks = [0.5] + [x + 0.5 for x in ax.get_xticks()]
        ax.set_xticks(xticks, minor=True)
        # Clean up the appearance
        ax.tick_params(axis='x', which='minor', length=3, width=1)

        # Change the colour of the boxes to Seaborn's 'pastel' palette
        colors = sns.color_palette('pastel')
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        # Colour of the median lines
        plt.setp(bp['medians'], color='k')

        # Check for statistical significance
        significant_combinations = []
        p_values = []

        # Check from the outside pairs of boxes inwards
        ls = list(range(1, len(data) + 1))
        combinations = [(ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))]
        for c in combinations:
            data1 = data[c[0] - 1]
            data2 = data[c[1] - 1]
            # Correct concatenation for the statistical test
            combined_data = np.concatenate((data1, data2))  # Note the double parentheses
            # Create a DataFrame suitable for pingouin's ANOVA
            groups = ['data1'] * len(data1) + ['data2'] * len(data2)  # Correctly repeat group labels
            df = pd.DataFrame({'score': combined_data, 'group': groups})
            # Perform the ANOVA
            aov = pg.anova(dv='score', between='group', data=df, detailed=True)
            # Append p-value and combination to lists if significant
            p_values.append(aov.at[0, 'p-unc'])
            if aov.at[0, 'p-unc'] < 0.05:  # Accessing p-value
                significant_combinations.append([c, aov.at[0, 'p-unc']])
        
        for i, (combination, p_val) in enumerate(zip(significant_combinations, p_values)):
            print(f"Significant difference between groups {combination[0]}: p-value = {p_val}")


        # Get info about y-axis
        bottom, top = ax.get_ylim()
        yrange = top - bottom

        # Significance bars
        for i, significant_combination in enumerate(significant_combinations):
            # Columns corresponding to the datasets of interest
            x1 = significant_combination[0][0]
            x2 = significant_combination[0][1]
            # What level is this bar among the bars above the plot?
            level = len(significant_combinations) - i
            # Plot the bar
            bar_height = (yrange * 0.08 * level) + top
            bar_tips = bar_height - (yrange * 0.02)
            plt.plot(
                [x1, x1, x2, x2],
                [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
            # Significance level
            p = significant_combination[1]
            if p < 0.001:
                sig_symbol = '***'
            elif p < 0.01:
                sig_symbol = '**'
            elif p < 0.05:
                sig_symbol = '*'
            text_height = bar_height + (yrange * 0.01)
            plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', c='k')

        # Adjust y-axis
        bottom, top = ax.get_ylim()
        yrange = top - bottom
        ax.set_ylim(bottom - 0.02 * yrange, top)

        # Annotate sample size below each box
        for i, dataset in enumerate(data):
            sample_size = len(dataset)
            ax.text(i + 1, bottom, fr'n = {sample_size}', ha='center', size='x-small')


        save_path = folder_path + f"/images/{method_name}/stats_comparison_nb_direction_{nb_direction}_{patient_path}_{method_name}_{data_type}_{preprocessing_type}.png"
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path)
        print(f"Plot saved as PNG: {save_path}")
        plt.close()
        return

    print("Creating box-and-whisker plot...")
    if method_name =="DTI":
        title = f"Comparison of {data_type} Data for {patient_path} (Method: {method_name})"
        box_and_whisker(filtered_data, title, data_type, labels,method_name,data_type)
    else :
        title = f"Comparison of {data_type} Data for {patient_path} (Method: {method_name})"
        box_and_whisker(filtered_data, title, data_type, labels ,method_name,data_type)
    print("Box-and-whisker plot created.")

    return

#########################################################################################################################################
    




#method ="DTI"
method ="noddi"
#method ="MF"


method_list = ["DTI","noddi","MF"]

parameters_dti = ["FA","MD","RD","AD"]
parameters_noddi = ["fbundle","fextra", "fintra","fiso", "icvf", "odi"]#mu
parameters_mf = ["fvf_f0","fvf_f1","fvf_tot", "frac_csf","frac_f0","frac_f1","DIFF_ex_f0","DIFF_ex_f1","DIFF_ex_tot"]

atlas_name = "JHU-ICBM-labels-1mm"

atlas_values = [5]
roi_name ="body_of_corpus_callosum"

tract_dictionary = {
#    1: "unclassified",
#    2: "middle cerebellar peduncle",
#    3: "pontine crossing tract",
    4: "genu of corpus callosum",
    5: "body of corpus callosum",
    6: "splenium of corpus callosum",
#    7: "fornix",
#    8: "corticospinal tract r",
#    9: "corticospinal tract l",
#    10: "medial lemniscus r",
#    11: "medial lemniscus l",
#    12: "inferior cerebellar peduncle r",
#    13: "inferior cerebellar peduncle l",
#    14: "superior cerebellar peduncle r",
#    15: "superior cerebellar peduncle l",
#    16: "cerebral peduncle r",
#    17: "cerebral peduncle l",
    18: "anterior limb of internal capsule r",
    19: "anterior limb of internal capsule l",
    20: "posterior limb of internal capsule r",
    21: "posterior limb of internal capsule l",
#    22: "retrolenticular part of internal capsule r",
#    23: "retrolenticular part of internal capsule l",
#    24: "anterior corona radiata r",
#    25: "anterior corona radiata l",
#    26: "superior corona radiata r",
#    27: "superior corona radiata l",
#    28: "posterior corona radiata r",
 #   29: "posterior corona radiata l",
 #   30: "posterior thalamic radiation", # appears twice
 #   31: "sagittal stratum", # appears twice
 #   32: "external capsule r",
 #   33: "external capsule l",
 #   34: "cingulum", # cingulum appears four times (36 to 39)
 #   35: "cingulum",
 #   36: "cingulum",
 #   37: "cingulum",
 #   38: "fornix", # fornix appears twice (7 and 41)
 #   39: "fornix",
 #   40: "superior longitudinal fasciculus r",
  #  41: "superior longitudinal fasciculus l",
  #  42: "superior fronto-occipital fasciculus", # appears twice (44 and 45)
  #  43: "superior fronto-occipital fasciculus",
  #  44: "uncinate fasciculus r",
  #  45: "uncinate fasciculus l",
  #  46: "tapetum r",
  #  47: "tapetum l"
}





#compare_nb_bundles("DTI", "FA",1)
#compare_nb_bundles("DTI", "FA",2)
#compare_nb_bundles("DTI", "FA",3)



for i in range(1001,1004):
    patient_path = f"sub-{i}"

    reduced_bval = 1000
    reduced_nb=[16,32,40]

    plot_values_bvals(patient_path, method)

    #plot_values_bvals_dti(patient_path, "DTI")

    #plot_values_pair_DTI(patient_path, "DTI")

    plot_values_reduced(patient_path, method,1000)
    
    plot_values_reduced(patient_path, method,3000)

    plot_values_reduced(patient_path, method,10000)

    plot_values_reduced(patient_path, method,5000)

    #plot_values_reduced_ROI(patient_path, method,5000,[5],atlas_name,"genu of corpus callosum")


    # calculate_and_plot_comparison_reduced_boxplot_mse(patient_path,reduced_bval, reduced_nb, method, show=False, save_as_png=True)
    
    #calculate_and_plot_hist_mse_multi_methods(patient_path, method_list, preproc_raw="preproc", show=False, save_as_png=True)

    #calculate_and_plot_boxplot_mse_multi_methods_subplot(patient_path, method_list, prepoc_raw="preproc", show=False, save_as_png=True)

    #visualize_method_MSE_by_bvalue(patient_path, method, prepoc_raw="preproc", show=False, save_as_png=True)

    #visualize_method_MSE_by_bvalue_pair(patient_path, method, prepoc_raw="preproc", show=False, save_as_png=True)

    #calculate_and_plot_boxplot_mse_multi_methods_1_fig(patient_path, method_list, prepoc_raw="preproc", show=False, save_as_png=True)

    #calculate_and_plot_boxplot_mse_brain_white_matter(patient_path, method, prepoc_raw="preproc", show=False, save_as_png=False)

    #calculate_and_plot_boxplot_mse_multi_b_values_1_fig_ROI(patient_path, method, atlas_name, atlas_values, roi_name, prepoc_raw="preproc", show=False, save_as_png=True)

    for parameter in parameters_mf:
        
        pass
        #plot_slices_boxplot_roi(patient_path, atlas_name, atlas_values, roi_name, method, parameter, preprocessing_type="preproc", show=False, save_as_png=True)
        #compare_MRI_bvalues_statistically(patient_path, method, parameter, preprocessing_type="preproc", show=False, save_as_png=True)
        #plot_and_statistical_comparison_slices_reduced(patient_path, method, parameter,reduced_bval,reduced_nb, show=False, save_as_png=True)
        #plot_and_statistical_comparison_slices_reduced_ROI(patient_path,atlas_name, atlas_values, roi_name, method, parameter,reduced_bval,reduced_nb, show=False, save_as_png=True)
        #compare_MRI_bvalues_pair_statistically(patient_path, method, parameter, preprocessing_type="preproc", show=False, save_as_png=True)
        #compare_MRI_bvalues_ROI_statistically(patient_path,atlas_name, atlas_values, roi_name, method, parameter, show=False, save_as_png=True)
        #compare_MRI_bvalues_pair_ROI_statistically(patient_path,atlas_name, atlas_values, roi_name, method, parameter, show=False, save_as_png=True)
        #for key, value in tract_dictionary.items():
        #    compare_MRI_bvalues_pair_ROI_statistically(patient_path,atlas_name, [key], value, method, parameter, show=False, save_as_png=True)

        #    compare_MRI_bvalues_ROI_statistically(patient_path,atlas_name, [key], value, method, parameter, show=False, save_as_png=True)

        #    plot_and_statistical_comparison_slices_reduced_ROI(patient_path,atlas_name, [key], value, method, parameter,reduced_bval,reduced_nb, show=False, save_as_png=True)
        
        
        #compare_MRI_bvalues_to_csv(patient_path, method, parameter, preprocessing_type="preproc", save_csv = True)




#for key, value in tract_dictionary.items():
#    calculate_mse_multi_methods_write_csv_ROI(range(1001, 1011), method_list,atlas_name, [key], value)
#    plot_mse_comparison(f"/auto/home/users/g/d/gdeside/Mine/results/mse/mse_values_comparison_method_{value}.csv",f"comparison_{value}",save_as_png=True,show=False)

#calculate_mse_multi_methods_write_csv(range(1001, 1011), ["DTI","noddi", "MF"])
#plot_mse_comparison("/auto/home/users/g/d/gdeside/Mine/results/mse/mse_values_comparison_DTI_pair.csv","DTI_pair",save_as_png=True,show=False)
#calculate_method_MSE_by_bvalue_pair_write_csv(range(1001, 1011), "DTI", prepoc_raw="preproc")

#calculate_method_MSE_by_bvalue_write_csv(range(1001, 1011), "DTI", prepoc_raw="preproc")
#plot_mse_comparison("/auto/home/users/g/d/gdeside/Mine/results/mse/mse_values_comparison_DTI_by_bval.csv","DTI_by_bval",save_as_png=True,show=False)




#calculate_method_MSE_by_bvalue_pair_write_csv(range(1001, 1011), "noddi", prepoc_raw="preproc")
#plot_mse_comparison("/auto/home/users/g/d/gdeside/Mine/results/mse/mse_values_comparison_noddi_pair.csv","noddi_pair",save_as_png=True,show=False)
#calculate_method_MSE_by_bvalue_write_csv(range(1001, 1011), "noddi", prepoc_raw="preproc")
#plot_mse_comparison("/auto/home/users/g/d/gdeside/Mine/results/mse/mse_values_comparison_noddi_by_bval.csv","noddi_by_bval",save_as_png=True,show=False)




#calculate_method_MSE_by_bvalue_write_csv(range(1001, 1011), "MF", prepoc_raw="preproc")

#plot_mse_comparison("/auto/home/users/g/d/gdeside/Mine/results/mse/mse_values_comparison_MF_pair.csv","MF_pair",save_as_png=True,show=False)


#calculate_method_MSE_by_bvalue_pair_write_csv(range(1001, 1011), "MF", prepoc_raw="preproc")


reduced_bval = 5000
reduced_nb=[32, 64, 100]
#for parameter in parameters_dti:
#    compare_nb_bundles("DTI", parameter,1)
#    compare_nb_bundles("DTI", parameter,2)
#    compare_nb_bundles("DTI", parameter,3)
#    calculate_reduced_write_csv(range(1001, 1011), "DTI", parameter, reduced_bval, reduced_nb)
#    calculate_bval_pair_write_csv(range(1001, 1011), "DTI", parameter)

#for parameter in parameters_dti:
    #calculate_bval_write_csv(range(1001, 1011), "DTI", parameter)
    #calculate_bval_pair_write_csv(range(1001, 1011), "DTI", parameter)

    #calculate_bval_write_csv_ROI(range(1001, 1011), "DTI", parameter, atlas_name, atlas_values, roi_name)

    #calculate_bval_pair_write_csv_ROI(range(1001, 1011), "DTI", parameter,atlas_name, atlas_values, roi_name)
     
    


#for parameter in parameters_noddi:
    #compare_nb_bundles("noddi", parameter,1)
    #compare_nb_bundles("noddi", parameter,2)
    #compare_nb_bundles("noddi", parameter,3)

    #compare_nb_bundles("noddi", parameter,1)
    #compare_nb_bundles("noddi", parameter,2)
    #compare_nb_bundles("noddi", parameter,3)
    #calculate_bval_pair_write_csv(range(1001, 1011), "noddi", parameter)
#    calculate_bval_write_csv(range(1001, 1011), "noddi", parameter)


#for parameter in parameters_mf:
#    calculate_bval_write_csv(range(1001, 1011), "MF", parameter)
#    calculate_bval_pair_write_csv(range(1001, 1011), "MF", parameter)


