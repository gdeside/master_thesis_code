# Import necessary libraries and modules for handling math operations, file paths,
# arguments from the command line, and date and time.
from __future__ import division
import argparse
import math
import os
import datetime
import numpy as np
from itertools import combinations
import copy

# Import modules specific for q-space sampling and visualization, and for plotting.
from qspace.sampling import multishell as ms
from qspace.visu import visu_points
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, art3d

from qspace.sampling import multishell as ms

from mpl_toolkits.mplot3d import Axes3D  # Even though not used directly, this import is necessary
from matplotlib import colors
from matplotlib.patches import Patch


def find_unique_closest_points(points, bvecs, num_points=16):
    closest_points = np.zeros((num_points, bvecs.shape[1]))  # Initialize array for closest points
    used_indices = set()  # Keep track of used indices to ensure uniqueness

    for i, point in enumerate(points[:num_points]):
        # Make Z-coordinate positive for comparison point
        modified_point = np.copy(point)
        modified_point[2] = abs(modified_point[2])
        
        min_distance = np.inf
        closest_index = -1

        for j, bvec in enumerate(bvecs):
            if j in used_indices:
                continue  # Skip if this bvec is already used

            # Make Z-coordinate positive for bvec
            modified_bvec = np.copy(bvec)
            modified_bvec[2] = abs(modified_bvec[2])

            # Calculate Euclidean distance with modified Z-coordinate
            distance = np.linalg.norm(modified_bvec - modified_point)
            if distance < min_distance:
                min_distance = distance
                closest_index = j

        if closest_index != -1:  # Found a closest point that hasn't been used
            closest_points[i] = bvecs[closest_index]  # Assign the original bvec, not modified
            used_indices.add(closest_index)  # Mark this index as used

    return closest_points

def compute_cost(bvecs, num_shells, points_per_shell, weights):
    bvecs_squeezed = bvecs.flatten()  # Flatten to fit your cost function's expected input format
    cost = ms.cost(bvecs_squeezed, num_shells, points_per_shell, weights, antipodal=True)
    return cost


def total_min_distance_after_rotation(angle_degrees, points, bvecs):
    angle_radians = np.radians(angle_degrees)  # Convert angle from degrees to radians
    modified_points = np.copy(points)
    modified_bvecs = np.copy(bvecs)

    # Set Z coordinates to absolute values
    modified_points[:, 2] = np.abs(modified_points[:, 2])
    modified_bvecs[:, 2] = np.abs(modified_bvecs[:, 2])

    used_indices = set()  # To track which bvecs have been used
    total_min_distance = 0

    for point in modified_points:
        rotated_point = rotate_around_z(point, angle_radians)
        min_distance = np.inf
        min_index = None
        for i, bvec in enumerate(modified_bvecs):
            if i not in used_indices:
                distance = np.linalg.norm(bvec - rotated_point)
                if distance < min_distance:
                    min_distance = distance
                    min_index = i
        
        # Add the minimum distance found if it's valid
        if min_index is not None:
            total_min_distance += min_distance
            used_indices.add(min_index)  # Mark this bvec as used

    return total_min_distance

def rotate_around_z(vector, angle_radians):
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians), 0],
        [np.sin(angle_radians), np.cos(angle_radians), 0],
        [0, 0, 1]
    ])
    return np.dot(rotation_matrix, vector)

def find_optimal_rotation(points, bvecs):
    best_distance = np.inf
    best_angle = 0
    for angle in range(0, 360, 10):  # Incrementing by 10 degrees
        current_distance = total_min_distance_after_rotation(angle, points, bvecs)
        if current_distance < best_distance:
            best_distance = current_distance
            best_angle = angle
    return best_angle

#####################################################################################################################################


bvalue = 5000
nb_points_lst = [64]

start_end_index_dic = {
    0 : [0,40] , 
    1000 : [40,104] ,
    3000 : [104,168] ,
    5000 : [168,296] ,
    10000 : [296,552] ,
}

start_index = start_end_index_dic[bvalue][0]
end_index   = start_end_index_dic[bvalue][1]


####################################################################################################################################
for nb_points in nb_points_lst:
    points_per_shell = [nb_points]
    nb_shells = len(points_per_shell)
    K = np.sum(points_per_shell)
    print(f"Number of shells (nb_shells): {nb_shells}, Total number of points (K): {K}")
        
    # Set up groups of shells and their coupling weights for the sampling scheme.
    shell_groups = [[i] for i in range(nb_shells)]
    shell_groups.append(range(nb_shells))
    alphas = np.ones(len(shell_groups))
    print(f"Shell groups: {shell_groups}, Alphas: {alphas}")

    # Compute weights based on the number of shells and specified points per shell.
    print("Computing weights for each shell group...")
    weights = ms.compute_weights(nb_shells, points_per_shell, shell_groups, alphas)
    print(f"Weights computed: {weights}")

    # Compute the optimized sampling scheme based on the above parameters.
    print("Optimizing sampling scheme...")
    points = ms.optimize(nb_shells, points_per_shell, weights, max_iter=1000)
    print("Optimization completed. Points generated.")



    fig = plt.figure(figsize=[10, 8])
    ax = fig.add_subplot(111, projection='3d')


    ax.scatter(points[:, 0], points[:, 1], points[:, 2], depthshade=True, c="b", label='optimized schema')



    # Setting the aspect ratio to 'equal' for a spherical representation
    ax.set_box_aspect([1, 1, 1])  # Or: ax.set_aspect('equal') for older versions of matplotlib

    # Axis labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #ax.set_title('3D Visualization of b-vectors')
    ax.legend(fontsize=20)

        
        
    # Save the plot to a JPEG file
    output_directory = os.getcwd()+f"/{bvalue}/"  # Define your output directory; adjust as needed
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)  # Create the directory if it does not exist

    output_filename = os.path.join(output_directory, f"{bvalue}_optimal_points.jpeg")  # Construct the full file path
    plt.savefig(output_filename, format='jpeg', dpi=300)  # Save the figure as a JPEG

    plt.close()  # Close the plot to free up memory




    ####################################################################################################################################

    # Loop for multiple subjects
    for sub_num in range(1001, 1002):
        # Update subject variable for each iteration
        subject = f"sub-{sub_num}"
        data_path = f"/CECI/proj/pilab/PermeableAccess/guillaume_Jd5dhd9vBkTz9m/HCP/subjects/{subject}/dMRI/preproc/"
        bvecs_file = f"{subject}_dmri_preproc.bvec"
        bvals_file = f"{subject}_dmri_preproc.bval"

        # Loading b-vectors and b-values
        print(f"\nLoading data for subject {subject}...")
        bvecs = np.loadtxt(data_path + bvecs_file)
        bvecs = bvecs.T
        bvals = np.loadtxt(data_path + bvals_file)
        print("Data loaded successfully.")
        print(f"Shape of bvecs: {bvecs.shape}")
        print(f"Shape of bvals: {bvals.shape}")
        print(f"Type of bvecs after transposition: {type(bvecs)}")


        # Analyzing b-values
        print("\nAnalyzing b-values distribution:")
        target_values = [0, 1000, 3000, 5000, 10000]
        margin = 200
        np_points_original = []
        bvecs_by_bval = []

        for value in target_values:
            count = np.sum((bvals >= (value - margin)) & (bvals <= (value + margin)))
            np_points_original.append(count)
            print(f"-> {count} bvals within ±{margin} of {value}")

        # Sorting bvecs and bvals
        print("\nSorting bvecs and bvals...")
        sorted_indices = np.argsort(bvals)
        sorted_bvals = bvals[sorted_indices]
        sorted_bvecs = bvecs[sorted_indices,: ]

        weighted_bvecs = [bvec * bval for bvec, bval in zip(sorted_bvecs, sorted_bvals)]
        weighted_bvecs = np.array(weighted_bvecs)
        print("Sorting completed.")

        print(f"Shape of sorted bvecs: {np.shape(sorted_bvecs)}")


        norm = colors.Normalize(vmin=np.min(bvals), vmax=np.max(bvals))
        cmap = plt.cm.jet  # You can choose another colormap (e.g., plt.cm.hot, plt.cm.cool)

        fig = plt.figure(figsize=[10, 8])
        ax = fig.add_subplot(111, projection='3d')

        # Plot each bvec, colored according to its bval using scatter
        scatter_colors = [cmap(norm(bval)) for bval in sorted_bvals]  # Create a list of colors for each point
        ax.scatter(weighted_bvecs[:, 0], weighted_bvecs[:, 1], weighted_bvecs[:, 2], c=scatter_colors, depthshade=True)

        #ax.scatter(points[:, 0], points[:, 1], points[:, 2],c="black")

        # Set the limits for x, y, z axis
        ax.set_xlim([-11000, 11000])
        ax.set_ylim([-11000, 11000])
        ax.set_zlim([-11000, 11000])

        # Setting the aspect ratio to 'equal' for a spherical representation
        ax.set_box_aspect([1, 1, 1])  # Or: ax.set_aspect('equal') for older versions of matplotlib

        # Axis labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        #ax.set_title('3D Visualization of b-vectors')

        # Add color bar representing b-values
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(sorted_bvals)  # Make sure to use sorted bvals
        cbar = plt.colorbar(mappable, shrink=0.5, aspect=5, label='b-value') 

        # Save the plot to a JPEG file
        output_directory = os.getcwd()+f"/{subject}/{bvalue}/"  # Define your output directory; adjust as needed
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)  # Create the directory if it does not exist

        output_filename = os.path.join(output_directory, f"{subject}_bvecs_visualization.jpeg")  # Construct the full file path
        plt.savefig(output_filename, format='jpeg', dpi=300)  # Save the figure as a JPEG

        plt.close()  # Close the plot to free up memory

    #####################################################################################################################################

        points_b5000 = points

        bvecs_b5000 = bvecs[start_index:end_index,: ]

        print(len(points_b5000),len(bvecs_b5000))

        # Create the plot
        fig = plt.figure(figsize=[10, 8])
        ax = fig.add_subplot(111, projection='3d')

        # Plot bvecs_b5000
        ax.scatter(bvecs_b5000[:, 0], bvecs_b5000[:, 1], bvecs_b5000[:, 2], c='blue', marker='o', label='original schema')

        # Plot points_b5000
        ax.scatter(points_b5000[:, 0], points_b5000[:, 1], points_b5000[:, 2], c='red', marker='^', label='optimized schema')

        # Setting the aspect ratio to 'equal' makes the scale the same along each axis
        ax.set_box_aspect([1,1,1])  # Or: ax.set_aspect('equal') for older versions of matplotlib

        # Labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        #ax.set_title('3D Visualization of bvecs_b5000 and points_b5000')
        ax.legend(fontsize=20)

        output_filename = os.path.join(output_directory, f"{subject}_bvecs_points_{bvalue}_{nb_points}.jpeg")  # Construct the full file path
        plt.savefig(output_filename, format='jpeg', dpi=300)  # Save the figure as a JPEG

        plt.close()  # Close the plot to free up memory
    ##############################################################################################################

        # Points and bvecs should be your data arrays
        optimal_angle = find_optimal_rotation(points, bvecs)
        print(f"Optimal rotation angle around the Z-axis: {optimal_angle} radians")

        # Rotate points with the optimal angle
        rotated_points = np.array([rotate_around_z(point, optimal_angle) for point in points])

        # Set up the 3D plot
        fig = plt.figure(figsize=[10, 8])
        ax = fig.add_subplot(111, projection='3d')

        # Plot original bvecs
        ax.scatter(bvecs[:, 0], bvecs[:, 1], abs(bvecs[:, 2]), color='blue', label='Original schema')

        # Plot rotated points
        ax.scatter(rotated_points[:, 0], rotated_points[:, 1], abs(rotated_points[:, 2]), color='red', label='Rotated schema')

        # Setting labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        #ax.set_title('3D Visualization of bvecs and Rotated Points')
        ax.legend(fontsize=20)

        # Equal aspect ratio to ensure the scales are the same along each axis
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_box_aspect([1,1,1])

        # Save the figure
        plt.savefig(f'{output_directory}/{subject}_optimized_rotation_plot_{bvalue}_{nb_points}.jpeg', format='jpeg', dpi=300)
        plt.close()  # Close the plot to free up memory

    ##############################################################################################################


        # Find the 64 unique closest points
        closest_points = find_unique_closest_points(rotated_points, bvecs_b5000, nb_points)
        print("Unique closest points found:\n", closest_points)


        fig = plt.figure(figsize=[10, 8])
        ax = fig.add_subplot(111, projection='3d')

        # Plot bvecs_b5000
        ax.scatter(closest_points[:, 0], closest_points[:, 1], abs(closest_points[:, 2]), c='blue', marker='o', label='closest points')

        # Plot points_b5000
        ax.scatter(points_b5000[:, 0], points_b5000[:, 1], abs(points_b5000[:, 2]), c='red', marker='^', label='optimized schema')

        # Setting the aspect ratio to 'equal' makes the scale the same along each axis
        ax.set_box_aspect([1,1,1])  # Or: ax.set_aspect('equal') for older versions of matplotlib

        # Labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        #ax.set_title('3D Visualization of bvecs_b5000 and points_b5000')
        ax.legend(fontsize=20)

        output_filename = os.path.join(output_directory, f"{subject}_bvecs_visualization__{bvalue}_{nb_points}_closest.jpeg")  # Construct the full file path
        plt.savefig(output_filename, format='jpeg', dpi=300)  # Save the figure as a JPEG

        plt.close()  # Close the plot to free up memory


    ##################################################################################################################################

        nb_shells = len(target_values)

        print(f"nb_shells type: {type(nb_shells)}, nb_shells value: {nb_shells}")
        assert isinstance(nb_shells, int), "nb_shells should be an integer"

        shell_groups = [[i] for i in range(nb_shells)] + [list(range(nb_shells))]
        alphas = np.ones(len(shell_groups))

        weights_orignal = ms.compute_weights(nb_shells, [40, 64, 64, 128, 256], shell_groups, alphas)

        if bvalue == 0:
            nb_points_lst = [nb_points, 64, 64, 128, 256]
        elif bvalue == 1000:
            nb_points_lst = [40, nb_points, 64, 128, 256]
        elif bvalue == 3000:
            nb_points_lst = [40, 64, nb_points, 128, 256]
        elif bvalue == 5000:
            nb_points_lst = [40, 64, 64, nb_points, 256]
        elif bvalue == 10000:
            nb_points_lst = [40, 64, 64, 128, nb_points]
        else:
            nb_points_lst = [40, 64, 64, 128, 256]


        weights_closest = ms.compute_weights(nb_shells, nb_points_lst, shell_groups, alphas)


        # Process closest_points_full
        end_index_bis = start_index + nb_points
        removal_start_index = end_index_bis
        removal_end_index = end_index

        print("end_index:", end_index_bis)
        print("removal_start_index:", removal_start_index)
        print("removal_end_index:", removal_end_index)


        closest_points_full = copy.deepcopy(sorted_bvecs)
        closest_points_full[start_index:end_index_bis, :] = closest_points
        closest_points_full = np.delete(closest_points_full, slice(removal_start_index, removal_end_index), axis=0)

        print("Dimensions of closest_points_full:", closest_points_full.shape)

        # Calculate costs
        cost_points = compute_cost(sorted_bvecs, nb_shells, [40, 64, 64, 128, 256], weights_orignal)
        cost_closest = compute_cost(closest_points_full, nb_shells, nb_points_lst, weights_closest)

        # Print and compare costs
        print(f"Cost for original points: {cost_points}")
        print(f"Cost for closest points: {cost_closest}")

        # Determine which set has the lowest cost and print the corresponding set
        if cost_points < cost_closest:
            print("Set of points with the lowest cost is original points:")
            print(sorted_bvecs)  # Or format as needed
        else:
            print("Set of points with the lowest cost is closest points:")
            print(closest_points_full)  # Or format as needed


    ####################################################################################################################################

        output_file_path = f'{output_directory}{subject}_closest_points_full_b{bvalue}_{nb_points}.txt'

        # Write the array to the text file
        np.savetxt(output_file_path, closest_points_full, fmt='%f', delimiter=',')

        print(f"Closest points have been saved to {output_file_path}")