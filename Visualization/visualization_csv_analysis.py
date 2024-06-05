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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import csv

def plot_mse_comparison_zoom_noddi(csv_path, name,xlabel, save_as_png=True, show=False):
    # Read the data
    data = pd.read_csv(csv_path)

    # Prepare the path for saving the image
    images_path = os.path.join(os.path.dirname(os.getcwd()), "images/mse")

    # Drop the patient ID column for analysis
    data.drop(columns='Patient ID', inplace=True)

    # Set up the main plot
    fig, ax_main = plt.subplots(figsize=(12, 8))
    sns.boxplot(data=data, ax=ax_main, showfliers=False)
    ax_main.set_ylabel('MSE', fontsize=14)  # Increase fontsize for y-label
    ax_main.set_xlabel(xlabel, fontsize=14)  # Increase fontsize for x-label
    ax_main.tick_params(axis='both', labelsize=12)  # Increase fontsize for the axis ticks
    
    # Apply minimal theme
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)
    ax_main.spines['left'].set_visible(False)
    ax_main.spines['bottom'].set_visible(False)
    ax_main.grid(False)
    ax_main.set_facecolor('white')
    ax_main.yaxis.grid(True, color='lightgrey', linestyle='--', linewidth=0.7)  # Add light gridlines only on y-axis

    # Set up the inset axes for a specific method (e.g., noddi)
    ax_inset = inset_axes(ax_main, width="30%", height="30%", loc="upper right")

    # Create a boxplot for the "noddi" method data within the inset axes
    noddi_data = data['noddi'].dropna()  # Extract "noddi" data and remove NaN values
    sns.boxplot(y=noddi_data, ax=ax_inset, showfliers=False, color='purple')  # Set color for inset boxplot
    ax_inset.tick_params(axis='y', labelsize=12)  # Adjust the fontsize for inset ticks
    ax_inset.set_ylabel('MSE', fontsize=12)  # Increase fontsize for y-label
    ax_inset.set_xlabel('noddi', fontsize=12) 

    # Apply minimal theme to inset
    ax_inset.spines['top'].set_visible(False)
    ax_inset.spines['right'].set_visible(False)
    ax_inset.spines['left'].set_visible(False)
    ax_inset.spines['bottom'].set_visible(False)
    ax_inset.grid(False)
    ax_inset.set_facecolor('white')
    ax_inset.yaxis.grid(True, color='lightgrey', linestyle='--', linewidth=0.7)  # Add light gridlines only on y-axis

    # Optional: Set titles or additional text
    # ax_main.set_title('MSE Comparison Across Methods', fontsize=16)  # Set the title for main axes
    # ax_inset.set_title("Noddi:", fontsize=12)  # Set title for inset

    # Save or show the plot
    if save_as_png:
        plt.savefig(os.path.join(images_path, f"boxplot_mse_white_matter_brain_all_patients_{name}.png"), bbox_inches='tight')
    if show:
        plt.show()

    plt.close()


def plot_mse_comparison(csv_path, name, xlabel, save_as_png=True, show=False):
    # Read the data
    data = pd.read_csv(csv_path)

    images_path = os.path.join(os.path.dirname(os.getcwd()), "images/mse")
    
    # Drop the patient ID column for analysis
    data.drop('Patient ID', axis=1, inplace=True)

    # Create the boxplot
    fig, ax = plt.subplots(figsize=(12, 8))
    data.plot(kind='box', ax=ax, showfliers=False)
    
    # Set labels with increased font size
    ax.set_ylabel('MSE', fontsize=14)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', labelsize=14)
    
    # Apply minimal theme
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.grid(False)
    ax.set_facecolor('white')
    ax.yaxis.grid(True, color='lightgrey', linestyle='--', linewidth=0.7)  # Add light gridlines only on y-axis

    # Save or show the plot
    if save_as_png:
        plt.savefig(os.path.join(images_path, f"boxplot_mse_white_matter_brain_all_patients_{name}.png"), bbox_inches='tight')
    if show:
        plt.show()
    plt.close()



def read_and_plot_all_roi_mse(folder_path, startfile):
    """
    Reads all MSE CSV files from a specified folder, combines them, and creates a boxplot comparing MSE across all ROIs.

    Args:
    - folder_path (str): Path to the folder containing MSE CSV files for each ROI.
    - startfile (str): Starting string for filenames to include in the analysis.

    Returns:
    - None: Outputs a boxplot and saves it to a specified directory.
    """
    mse_data = pd.DataFrame()
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and f.startswith(startfile)]
    files.sort()  # Sort the files alphabetically

    # Loop through each sorted file
    for file in files:
        if file.endswith('.csv') and file.startswith(startfile):
            print(file)
            roi_name = roi_name = file.replace(startfile, '').replace('.csv', '').replace('_', ' ').replace(folder_path,'').title()
            file_path = os.path.join(folder_path, file)
            temp_df = pd.read_csv(file_path)
            temp_df['ROI'] = roi_name  # Add a column for the ROI
            mse_data = pd.concat([mse_data, temp_df])

    # Reset index after concatenation
    mse_data.reset_index(drop=True, inplace=True)

    # Create the boxplot
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=mse_data, x='ROI', y='MSE',showfliers=False)
    plt.xticks(rotation=90)  # Rotate the ROI names for better readability
    #plt.title(f'Comparison of MSE across different ROIs starting with {startfile}')
    plt.ylabel('MSE', fontsize=14)
    plt.xlabel('Region of Interest (ROI)',fontsize=14)
    plt.tight_layout()  # Adjust layout to make room for label rotation

    # Specify save path and save the plot
    save_folder = '../images/mse'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_path = os.path.join(save_folder, f'{startfile}_mse_comparison.png')
    plt.savefig(save_path)
    print(f"Plot saved as: {save_path}")

    # Display the plot
    plt.close()


#########################################################################################################################################

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



for key, value in tract_dictionary.items():
    
    #plot_mse_comparison(f"/auto/home/users/g/d/gdeside/Mine/results/mse/mse_values_comparison_method_{value}.csv",f"comparison_{value}",save_as_png=True,show=False)

    plot_mse_comparison_zoom_noddi(f"/auto/home/users/g/d/gdeside/Mine/results/mse/mse_values_comparison_method_{value}.csv", f"comparison_3_methods_{value}","Methods", save_as_png=True, show=False)




#read_and_plot_all_roi_mse("../results/mse","mse_DTI_")

#read_and_plot_all_roi_mse("../results/mse","mse_noddi_")

#read_and_plot_all_roi_mse("../results/mse","mse_MF_")


#plot_mse_comparison_zoom_noddi("/auto/home/users/g/d/gdeside/Mine/results/mse/mse_values_comparison_method.csv", "3_methods","Methods", save_as_png=True, show=False)



#plot_mse_comparison("/auto/home/users/g/d/gdeside/Mine/results/mse/mse_values_comparison_DTI_pair.csv","DTI_pair","shells used",save_as_png=True,show=False)
#plot_mse_comparison("/auto/home/users/g/d/gdeside/Mine/results/mse/mse_values_comparison_noddi_pair.csv","noddi_pair","shells used",save_as_png=True,show=False)
#plot_mse_comparison("/auto/home/users/g/d/gdeside/Mine/results/mse/mse_values_comparison_MF_pair.csv","MF_pair","shells used",save_as_png=True,show=False)

#plot_mse_comparison("/auto/home/users/g/d/gdeside/Mine/results/mse/mse_values_comparison_DTI_by_bval.csv","DTI_by_bval","shells used",save_as_png=True,show=False)
#plot_mse_comparison("/auto/home/users/g/d/gdeside/Mine/results/mse/mse_values_comparison_noddi_by_bval.csv","noddi_by_bval","shells used",save_as_png=True,show=False)
#plot_mse_comparison("/auto/home/users/g/d/gdeside/Mine/results/mse/mse_values_comparison_MF_by_bval.csv","MF_by_bval","shells used",save_as_png=True,show=False)


#plot_mse_comparison("/auto/home/users/g/d/gdeside/Mine/results/mse/mse_values_comparison_DTI_reduced_1000.csv","DTI_reduced_1000","number of directions",save_as_png=True,show=False)

#plot_mse_comparison("/auto/home/users/g/d/gdeside/Mine/results/mse/mse_values_comparison_noddi_reduced_1000.csv","NODDI_reduced_1000","number direction",save_as_png=True,show=False)
#plot_mse_comparison("/auto/home/users/g/d/gdeside/Mine/results/mse/mse_values_comparison_noddi_reduced_3000.csv","NODDI_reduced_3000","number direction",save_as_png=True,show=False)
#plot_mse_comparison("/auto/home/users/g/d/gdeside/Mine/results/mse/mse_values_comparison_noddi_reduced_5000.csv","NODDI_reduced_5000","number direction",save_as_png=True,show=False)
#plot_mse_comparison("/auto/home/users/g/d/gdeside/Mine/results/mse/mse_values_comparison_noddi_reduced_10000.csv","NODDI_reduced_10000","number direction",save_as_png=True,show=False)


#plot_mse_comparison("/auto/home/users/g/d/gdeside/Mine/results/mse/mse_values_comparison_MF_reduced_1000.csv","MF_reduced_1000","number of directions",save_as_png=True,show=False)
#plot_mse_comparison("/auto/home/users/g/d/gdeside/Mine/results/mse/mse_values_comparison_MF_reduced_3000.csv","MF_reduced_3000","number of directions",save_as_png=True,show=False)
#plot_mse_comparison("/auto/home/users/g/d/gdeside/Mine/results/mse/mse_values_comparison_MF_reduced_5000.csv","MF_reduced_5000","number of directions",save_as_png=True,show=False)
#plot_mse_comparison("/auto/home/users/g/d/gdeside/Mine/results/mse/mse_values_comparison_MF_reduced_10000.csv","MF_reduced_10000","number of directions",save_as_png=True,show=False)
