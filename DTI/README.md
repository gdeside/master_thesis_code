File: patient_dti_analysis.py

Author: Guillaume Francois Deside
Description

This file contains a set of functions for processing and analyzing Diffusion Tensor Imaging (DTI) data for individual patients. The main operations include loading DWI data, applying brain extraction, fitting a tensor model, calculating DTI metrics, and computing Mean Squared Error (MSE). The file also supports processing DTI data for specific b-values and pairs of b-values.
Functions
1. process_dti

python

def process_dti(patient_path, b0_threshold=60, bet_median_radius=2, bet_numpass=1, bet_dilate=2, prepoc_raw="preproc"):
    """
    Process Diffusion Tensor Imaging (DTI) data for an individual patient.

    This function performs several steps to process the DTI data of a given patient, including:
    1. Loading the DWI data, b-values, and b-vectors.
    2. Filtering the DWI data to select specific b-values.
    3. Applying brain extraction and generating a brain mask.
    4. Fitting a tensor model to the DWI data.
    5. Calculating and saving DTI metrics (FA, MD, RD, AD, eigenvectors, eigenvalues, RGB FA, and residuals).

    Parameters:
    - patient_path (str): The patient identifier.
    - b0_threshold (int, optional): B0 threshold for gradient table estimation. Default is 60.
    - bet_median_radius (int, optional): Median radius for brain extraction. Default is 2.
    - bet_numpass (int, optional): Number of passes for brain extraction. Default is 1.
    - bet_dilate (int, optional): Dilation factor for brain mask. Default is 2.
    - prepoc_raw (str, optional): Specify whether to use preprocessed ("preproc") or raw ("raw") data. Default is "preproc".

    Returns:
    - None
    """

Processes DTI data for an individual patient. The function performs the following steps:

    Loads the DWI data, b-values, and b-vectors.
    Filters the DWI data to select specific b-values (1000 and 3000).
    Applies brain extraction and generates a brain mask.
    Fits a tensor model to the DWI data.
    Calculates and saves DTI metrics such as Fractional Anisotropy (FA), Mean Diffusivity (MD), Radial Diffusivity (RD), Axial Diffusivity (AD), eigenvectors, eigenvalues, RGB FA, and residuals.

2. calculate_and_plot_dti_mse

python

def calculate_and_plot_dti_mse(patient_path, prepoc_raw="preproc", save_as_png=False, show=False):
    """
    Calculate and plot Mean Squared Error (MSE) for Diffusion Tensor Imaging (DTI).

    This function performs the following steps:
    1. Sets up file paths and directories.
    2. Loads the residuals from the DTI analysis.
    3. Calculates the MSE from the residuals.
    4. Prints statistical information about the MSE.
    5. Saves the MSE as a NIfTI file.
    6. Plots the MSE and optionally saves the plot as a PNG file.

    Parameters:
    - patient_path (str): The name of the patient.
    - prepoc_raw (str, optional): Specify whether to use preprocessed ("preproc") or raw ("raw") data. Default is "preproc".
    - save_as_png (bool, optional): Whether to save the plot as a PNG file. Default is False.
    - show (bool, optional): Whether to display the plot using plt.show(). Default is False.

    Returns:
    - None
    """

Calculates and plots the Mean Squared Error (MSE) for DTI. The function performs the following steps:

    Sets up file paths and directories.
    Loads the residuals from the DTI analysis.
    Calculates the MSE from the residuals.
    Prints statistical information about the MSE.
    Saves the MSE as a NIfTI file.
    Optionally saves the plot as a PNG file and/or displays the plot.

3. process_dti_b_values

python

def process_dti_b_values(patient_path, b0_threshold=60, bet_median_radius=2, bet_numpass=1, bet_dilate=2, prepoc_raw="preproc"):
    """
    Process Diffusion Tensor Imaging (DTI) for an individual patient for multiple b-values.

    This function performs the following steps for each specified b-value:
    1. Sets up file paths and directories.
    2. Loads the diffusion-weighted imaging (DWI) data, b-values, and b-vectors.
    3. Selects the data corresponding to the target b-value.
    4. Performs brain extraction using median Otsu.
    5. Constructs a gradient table.
    6. Fits a tensor model to the data.
    7. Calculates and saves the DTI metrics (FA, MD, RD, AD) and their color-coded representations.
    8. Saves the eigenvalues and eigenvectors of the tensor model.
    9. Predicts the diffusion signal and calculates the residuals.
    10. Saves the residuals.

    Parameters:
    - patient_path (str): The identifier of the patient.
    - b0_threshold (int): B0 threshold for gradient table estimation.
    - bet_median_radius (int): Median radius for brain extraction.
    - bet_numpass (int): Number of passes for brain extraction.
    - bet_dilate (int): Dilation factor for brain mask.
    - prepoc_raw (str): Specify whether to use preprocessed ("preproc") or raw ("raw") data.

    Returns:
    - None
    """

Processes DTI data for an individual patient for multiple b-values. The function performs the following steps for each specified b-value (1000 and 3000):

    Sets up file paths and directories.
    Loads the diffusion-weighted imaging (DWI) data, b-values, and b-vectors.
    Selects the data corresponding to the target b-value.
    Performs brain extraction using the median Otsu method.
    Constructs a gradient table.
    Fits a tensor model to the data.
    Calculates and saves the DTI metrics (FA, MD, RD, AD) and their color-coded representations.
    Saves the eigenvalues and eigenvectors of the tensor model.
    Predicts the diffusion signal and calculates the residuals.
    Saves the residuals.

4. calculate_and_plot_dti_mse_b_value

python

def calculate_and_plot_dti_mse_b_value(patient_path, prepoc_raw="preproc", save_as_png=False, show=True):
    """
    Calculate and plot Mean Squared Error (MSE) for Diffusion Tensor Imaging (DTI) at a specific b-value.

    This function performs the following steps:
    1. Sets up file paths and directories.
    2. Loads the residuals of the diffusion signal prediction.
    3. Calculates the Mean Squared Error (MSE) from the residuals.
    4. Saves the MSE as a NIfTI file.
    5. Plots the MSE and optionally saves the plot as a PNG file.

    Parameters:
    - patient_path (str): The identifier of the patient.
    - prepoc_raw (str): Specify whether to use preprocessed ("preproc") or raw ("raw") data.
    - show (bool): Whether to display the plot using plt.show(). Default is True.
    - save_as_png (bool): Whether to save the plot as a PNG file. Default is False.

    Returns:
    - None
    """

Calculates and plots the Mean Squared Error (MSE) for DTI at specific b-values (1000 and 3000). The function performs the following steps:

    Sets up file paths and directories.
    Loads the residuals of the diffusion signal prediction.
    Calculates the MSE from the residuals.
    Saves the MSE as a NIfTI file.
    Optionally saves the plot as a PNG file and/or displays the plot.

5. process_dti_bval_pairs

python

def process_dti_bval_pairs(patient_path, b0_threshold=60, bet_median_radius=2, bet_numpass=1, bet_dilate=2, prepoc_raw="preproc"):
    """
    Process Diffusion Tensor Imaging (DTI) for an individual patient using pairs of specific b-values.

    This function performs the following steps:
    1. Sets up file paths and directories.
    2. Reads b-values and b-vectors.
    3. Loads the DWI data.
    4. Iterates through pairs of specified b-values and processes the DTI data for each pair.
    5. Performs brain extraction using the median_otsu method.
    6. Fits the DTI model and computes various diffusion metrics (FA, MD, RD, AD).
    7. Saves the computed metrics as NIfTI files.
    8. Computes and saves residuals of the DTI model fit.

    Parameters:
    - patient_path (str): The identifier of the patient.
    - b0_threshold (int): B0 threshold for gradient table estimation. Default is 60.
    - bet_median_radius (int): Median radius for brain extraction. Default is 2.
    - bet_numpass (int): Number of passes for brain extraction. Default is 1.
    - bet_dilate (int): Dilation factor for brain mask. Default is 2.
    - prepoc_raw (str): Specify whether to use preprocessed ("preproc") or raw ("raw") data. Default is "preproc".

    Returns:
    - None
    """

Processes DTI data for an individual patient using pairs of specific b-values (1000 and 3000). The function performs the following steps:

    Sets up file paths and directories.
    Reads b-values and b-vectors.
    Loads the DWI data.
    Iterates through pairs of specified b-values and processes the DTI data for each pair.
    Performs brain extraction using the median Otsu method.
    Fits the DTI model and computes various diffusion metrics (FA, MD, RD, AD).
    Saves the computed metrics as NIfTI files.
    Computes and saves residuals of the DTI model fit.

6. calculate_and_plot_dti_mse_pair

python

def calculate_and_plot_dti_mse_pair(patient_path, prepoc_raw="preproc"):
    """
    Calculate and plot Mean Squared Error (MSE) for Diffusion Tensor Imaging (DTI) for pairs of specific b-values.

    This function performs the following steps for each pair of specific b-values:
    1. Sets up file paths and directories.
    2. Reads the residuals of the DTI model fit.
    3. Calculates the Mean Squared Error (MSE) from the residuals.
    4. Prints statistical information about the MSE (mean, median, standard deviation).
    5. Saves the MSE as a NIfTI file.

    Parameters:
    - patient_path (str): The identifier of the patient.
    - prepoc_raw (str): Specify whether to use preprocessed ("preproc") or raw ("raw") data. Default is "preproc".

    Returns:
    - None
    """

Calculates and plots the Mean Squared Error (MSE) for DTI for pairs of specific b-values (1000 and 3000). The function performs the following steps:

    Sets up file paths and directories.
    Reads the residuals of the DTI model fit.
    Calculates the MSE from the residuals.
    Prints statistical information about the MSE (mean, median, standard deviation).
    Saves the MSE as a NIfTI file.

Example Usage

The following loop processes and analyzes DTI data for a range of patients (sub-1001 to sub-1010):

python

for i in range(1001, 1011):
    patient_path = f"sub-{i:04d}"

    process_dti(patient_path, b0_threshold=60, bet_median_radius=2, bet_numpass=1, bet_dilate=2)
    calculate_and_plot_dti_mse(patient_path, prepoc_raw="preproc", save_as_png=False, show=False)

    process_dti_b_values(patient_path, b0_threshold=60, bet_median_radius=2, bet_numpass=1, bet_dilate=2)
    calculate_and_plot_dti_mse_b_value(patient_path, prepoc_raw="preproc", save_as_png=False, show=True)
    
    process_dti_bval_pairs(patient_path, b0_threshold=60, bet_median_radius=2, bet_numpass=1, bet_dilate=2, prepoc_raw="preproc")
    calculate_and_plot_dti_mse_pair(patient_path, prepoc_raw="preproc")

Notes

    Ensure that the necessary libraries (numpy, dipy, matplotlib) are installed.
    Adjust file paths and directories as needed for your local environment.
    The functions assume a specific directory structure for the patient data. Adjust the paths if your data is organized differently.