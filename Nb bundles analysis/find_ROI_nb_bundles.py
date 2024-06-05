from dipy.io.image import load_nifti
import numpy as np
import numpy.ma as ma
import os
import pandas as pd
import nibabel as nib

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


path_fod_peaks = "/auto/home/users/g/d/gdeside/Mine/Registration/mrtrix_msmt_fod_peaks.nii.gz"

subject_1007_atlas = "/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/sub-1007/reg/sub-1007_Atlas_JHU-ICBM-labels-1mm_InSubjectDWISpaceFrom_AP.nii.gz"


data_fod_peaks, affine_fod_peaks = load_nifti(path_fod_peaks)

brain_mask, _ = load_nifti(subject_1007_atlas)

print(np.shape(brain_mask),np.shape(data_fod_peaks))

results = []

    # Loop through each tract specified in the dictionary
for key, value in tract_dictionary.items():
    # Create a mask for the current tract
    brain_mask_mask = np.where(np.isin(brain_mask, key), False, True)
    expanded_mask = np.expand_dims(brain_mask_mask, axis=-1) 
    expanded_mask = np.repeat(expanded_mask, data_fod_peaks.shape[3], axis=-1)

    masked_data = ma.masked_array(data_fod_peaks, mask=expanded_mask)

    print(np.shape(masked_data))
    # Assuming we are interested in all voxels that are not masked, across all slices
    for i in range(masked_data.shape[0]):
        for j in range(masked_data.shape[1]):
            for k in range(masked_data.shape[2]):
                if not np.all(masked_data.mask[i, j, k]):  # Check if not fully masked
                    # Extract the first 9 values from the fourth dimension
                    data_values = masked_data[i, j, k, :9]
                    vector1 = data_values[0:3]
                    vector2 = data_values[3:6]
                    vector3 = data_values[6:9]

                    # Calculate the norm of each vector
                    norm1 = np.linalg.norm(vector1)
                    norm2 = np.linalg.norm(vector2)
                    norm3 = np.linalg.norm(vector3)

                    threshold = 0.10
                    main_directions = sum([norm1 > threshold, norm2 > threshold, norm3 > threshold])

                    # Append the results with norms
                    results.append({
                        "tract": value,
                        "i": i,
                        "j": j,
                        "k": k,
                        #"values": data_values.tolist(),
                        "nb direction": main_directions ,
                        "norms": [norm1, norm2, norm3]  # Include norms in the results
                    })
print(len(results))
csv_path = os.getcwd() + "/tract.csv"
# Convert the results list to DataFrame and write to CSV
df = pd.DataFrame(results)
df.to_csv(csv_path, index=False)


map_norm = np.zeros((np.shape(data_fod_peaks)[0],np.shape(data_fod_peaks)[1],np.shape(data_fod_peaks)[2],3))
map_nb_direction = np.zeros((np.shape(data_fod_peaks)[0],np.shape(data_fod_peaks)[1],np.shape(data_fod_peaks)[2]))
for i in range(len(df)):
    index_i = int(df.loc[i, "i"])
    index_j = int(df.loc[i, "j"])
    index_k = int(df.loc[i, "k"])
    norms = df.loc[i, "norms"]
    nb_direction = df.loc[i, "nb direction"]
    map_norm[index_i, index_j, index_k] = norms
    map_nb_direction[index_i, index_j, index_k] = nb_direction




# Create separate masks for 1, 2, and 3 directions
mask_1_dir = map_nb_direction == 1
mask_2_dir = map_nb_direction == 2
mask_3_dir = map_nb_direction == 3

affine = affine_fod_peaks  # This should be obtained from the original data set

# Convert boolean masks to Nifti1Image and save
mask_1_img = nib.Nifti1Image(mask_1_dir.astype(np.int8), affine)
nib.save(mask_1_img, os.path.join(os.getcwd(), 'mask_1_direction.nii.gz'))

mask_2_img = nib.Nifti1Image(mask_2_dir.astype(np.int8), affine)
nib.save(mask_2_img, os.path.join(os.getcwd(), 'mask_2_directions.nii.gz'))

mask_3_img = nib.Nifti1Image(mask_3_dir.astype(np.int8), affine)
nib.save(mask_3_img, os.path.join(os.getcwd(), 'mask_3_directions.nii.gz'))

print("Direction masks saved as NIfTI files.")

total_voxels = np.count_nonzero(map_nb_direction)

# Count the number of True values in each mask
count_1_dir = np.sum(mask_1_dir)
count_2_dir = np.sum(mask_2_dir)
count_3_dir = np.sum(mask_3_dir)

# Calculate the percentage for each
percent_1_dir = (count_1_dir / total_voxels) * 100
percent_2_dir = (count_2_dir / total_voxels) * 100
percent_3_dir = (count_3_dir / total_voxels) * 100

# Print the results
print(f"Percentage of voxels with 1 direction: {percent_1_dir:.2f}%")
print(f"Percentage of voxels with 2 directions: {percent_2_dir:.2f}%")
print(f"Percentage of voxels with 3 directions: {percent_3_dir:.2f}%")

# Create separate NIfTI images for each mask
# Note: You may want to ensure you're handling empty data scenarios appropriately

# For 1 direction
map_1_dir = np.where(mask_1_dir[..., None], map_norm, 0)  # Preserve RGB channels
img_1_dir = nib.Nifti1Image(map_1_dir, affine_fod_peaks)
nib.save(img_1_dir, os.path.join(os.getcwd(), 'map_norm_1_direction.nii.gz'))

# For 2 directions
map_2_dir = np.where(mask_2_dir[..., None], map_norm, 0)  # Preserve RGB channels
img_2_dir = nib.Nifti1Image(map_2_dir, affine_fod_peaks)
nib.save(img_2_dir, os.path.join(os.getcwd(), 'map_norm_2_directions.nii.gz'))

# For 3 directions
map_3_dir = np.where(mask_3_dir[..., None], map_norm, 0)  # Preserve RGB channels
img_3_dir = nib.Nifti1Image(map_3_dir, affine_fod_peaks)
nib.save(img_3_dir, os.path.join(os.getcwd(), 'map_norm_3_directions.nii.gz'))

# Optional: Saving the number of directions as a separate NIfTI file could also be informative
img_nb_direction = nib.Nifti1Image(map_nb_direction, affine_fod_peaks)
nib.save(img_nb_direction, os.path.join(os.getcwd(), 'number_of_directions.nii.gz'))


from scipy.ndimage import label
from skimage.measure import regionprops

# Step 1: Identify voxels with exactly 3 directions
binary_map = (map_nb_direction == 3)

# Step 2: Group neighboring voxels using connected-component labeling
labeled_map, num_features = label(binary_map)

# Retrieve properties for each region, specifically their coordinates
regions = regionprops(labeled_map)

# Step 3: Write the coordinates of each region to a text file
with open('regions_coordinates.txt', 'w') as file:
    for region in regions:
        coordinates = region.coords  # Get coordinates for each region
        file.write(f'Region {region.label} with {len(coordinates)} voxels:\n')
        for coord in coordinates:
            file.write(f'{coord[0]}, {coord[1]}, {coord[2]}\n')
        file.write('\n')  # Add a newline for better separation of regions

print(f'Output written to regions_coordinates.txt with {num_features} regions found.')


map_norm = np.clip(map_norm, 0, np.max(map_norm))  # Assuming max is reasonable for your data
map_norm = (map_norm / np.max(map_norm)) * 255  # Normalize and scale to 255
map_norm = map_norm.astype(np.uint8)  # Convert to unsigned byte

# Create a Nifti1Image, note that NIfTI RGB data is expected to be uint8
norm_img = nib.Nifti1Image(map_norm, affine_fod_peaks)

# Save the new NIfTI image
nib.save(norm_img, os.path.join(os.getcwd(), 'map_norm_rgb.nii.gz'))

result_final = []


for key, value in tract_dictionary.items():
    df_value = df[df["tract"]==value]
    result_final.append(
        { "tract": value,
          "key":key,
          "number of direction": df_value['nb direction'].value_counts().idxmax()
        }
    )

csv_path_final = os.getcwd() + "/tract_final.csv"
df_final = pd.DataFrame(result_final)
df_final.to_csv(csv_path_final, index=False)














