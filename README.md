# Master Thesis Code Repository

Welcome to the repository for my master thesis code. This repository contains various scripts and files used in the analysis and visualization of neuroimaging data. Below is a detailed overview of the contents organized by folders.

## Folder Structure

### master_thesis_code
- **Visualization**
  - **visualization_csv_analysis.py**: Script for analyzing CSV files and generating visualizations.
  - **visualization_analysis.py**: General visualization analysis script.
  - **submit_job_helper.py**: Helper script for submitting jobs.
  - **scatter_analysis.py**: Script for performing scatter plot analysis.
  - **visualization_T1_analysis.py**: T1 visualization analysis script.
  
- **Nb bundles analysis**
  - **find_ROI_nb_bundles.py**: Script to find regions of interest in neuroimaging bundles.
  - **submit_job_helper.py**: Helper script for submitting jobs.
  - **mrtrix_msmt_fod_peaks.nii.gz**: NIfTI file used in bundle analysis.

- **Registration**
  - **Registration.py**: Script for registration of neuroimaging data.
  - **JHU-ICBM-labels-1mm.nii.gz**: NIfTI file containing JHU ICBM labels.
  - **Jhu_lables.txt**: Text file with JHU labels information.
  - **submit_job_helper.py**: Helper script for submitting jobs.

- **DTI**
  - **patient_dti_analysis.py**: Script for patient DTI (Diffusion Tensor Imaging) analysis.
  - **.DS_Store**: System file, should be ignored.
  - **README.md**: This readme file.
  - **submit_job_helper.py**: Helper script for submitting jobs.

- **MF**
  - (Currently empty)

- **Write csv**
  - **write_csv_analysis.py**: Script for writing analysis results to CSV files.
  - **submit_job_helper.py**: Helper script for submitting jobs.

- **Statistical analysis**
  - **boxplot_MF_bval.R**: R script for generating boxplots for MF b-values.
  - **.DS_Store**: System file, should be ignored.
  - **boxplot_DTI_bval.R**: R script for generating boxplots for DTI b-values.
  - **boxplot_MF_pair.R**: R script for generating paired boxplots for MF.
  - **boxplot_DTI_direction.R**: R script for generating boxplots for DTI directions.
  - **boxplot_NODDI.R**: R script for generating boxplots for NODDI.
  - **boxplot_NODDI_bval.R**: R script for generating boxplots for NODDI b-values.
  - **boxplot_NODDI_direction.R**: R script for generating boxplots for NODDI directions.
  - **Read_file.R**: R script for reading files.
  - **boxplot_NODDI_pair.R**: R script for generating paired boxplots for NODDI.
  - **boxplot_MF_direction.R**: R script for generating boxplots for MF directions.

- **dictionaries**
  - **dictionary-fixedraddist_scheme-HCPMGH.mat**: Dictionary file for HCP MGH scheme.
  - **dictionary-fixedraddist_scheme-StLucGE.mat**: Dictionary file for St. Luc GE scheme.
  - **dictionary-hexagpack_scheme-StLucGE.mat**: Dictionary file for hexagonal packing scheme at St. Luc GE.
  - **dictionary-hexagpack_scheme-HCPMGH.mat**: Dictionary file for hexagonal packing scheme at HCP MGH.

- **NODDI**
  - (Currently empty)

- **microstructure_fingerprinting**
  - (Currently empty)

- **Number of direction**
  - **patient_dti.py**: Script for patient DTI analysis with a focus on the number of directions.
  - **submit_job_helper.py**: Helper script for submitting jobs.
  - **reduced_number_of_directions.py**: Script for analyzing reduced number of directions.
  - **patient_mf.py**: Script for patient MF (Microstructure Fingerprinting) analysis.
  - **patient_noddi.py**: Script for patient NODDI (Neurite Orientation Dispersion and Density Imaging) analysis.

## How to Use

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/master_thesis_code.git
   ```
2. **Navigate to Desired Directory**:
   ```bash
   cd master_thesis_code/Visualization
   ```
3. **Run Scripts**:
   Follow the instructions within each script to execute. For example:
   ```bash
   python visualization_csv_analysis.py
   ```

   or use the following instruction after modifying it

   ```bash
   python submit_job_helper.py
   ```
   


## Contributing

If you wish to contribute to this repository, please fork the repository and create a pull request with your changes. Ensure your code follows the project's coding standards and include appropriate documentation.


## Contact

For any inquiries or issues, please contact [guillaume.deside@hotmail.com](mailto:guillaume.deside@hotmail.com).

